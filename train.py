import argparse, json, math, os, random
from dataclasses import asdict
from contextlib import nullcontext
import numpy as np, torch, yaml
import torch.nn.functional as F
from tqdm import tqdm
from models import ModelConfig, build_model, count_parameters, diversity_loss
from tasks import TaskConfig, make_dataloaders

IGNORE_LABEL_ID = -100

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_config(path):
    raw = yaml.safe_load(open(path))
    return ModelConfig(**raw['model']), TaskConfig(**raw['task']), raw['train']

def validate_config(model_cfg, task_cfg):
    if model_cfg.max_seq_len < task_cfg.seq_len:
        raise ValueError(f'model.max_seq_len={model_cfg.max_seq_len} must be >= task.seq_len={task_cfg.seq_len}')
    if model_cfg.vocab_size < task_cfg.vocab_size:
        raise ValueError(f'model.vocab_size={model_cfg.vocab_size} must be >= task.vocab_size={task_cfg.vocab_size}')

def autocast_context(device, train_cfg):
    enabled = bool(train_cfg.get('amp', device.startswith('cuda')))
    if not enabled or not device.startswith('cuda'):
        return nullcontext()
    dtype_name = train_cfg.get('amp_dtype', 'bfloat16')
    dtype = torch.bfloat16 if dtype_name == 'bfloat16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)

def _stablemax_transform(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)

def log_stablemax(x, dim=-1):
    transformed = _stablemax_transform(x)
    return torch.log(transformed / transformed.sum(dim=dim, keepdim=True))

def stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID):
    valid_mask = labels.ne(ignore_index)
    safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    selected = torch.gather(logprobs, dim=-1, index=safe_labels.long().unsqueeze(-1)).squeeze(-1)
    return -torch.where(valid_mask, selected, torch.zeros_like(selected, dtype=selected.dtype)).to(logits.dtype)

def prediction_loss(out, y, loss_type='cross_entropy'):
    if y.ndim == 1:
        return F.cross_entropy(out['logits'], y)
    if 'token_logits' not in out:
        raise ValueError('sequence-label training requires model outputs with token_logits')
    logits = out['token_logits']
    if loss_type == 'stablemax_cross_entropy':
        token_loss = stablemax_cross_entropy(logits, y)
        valid_counts = y.ne(IGNORE_LABEL_ID).sum(dim=-1).clamp_min(1)
        return (token_loss.sum(dim=-1) / valid_counts).mean()
    if loss_type != 'cross_entropy':
        raise ValueError(f'unknown loss_type={loss_type!r}; expected cross_entropy or stablemax_cross_entropy')
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE_LABEL_ID)

def exact_correct_from_output(out, y):
    if y.ndim == 1:
        pred = out['logits'].argmax(dim=-1)
        return pred.eq(y)
    pred = out['token_logits'].argmax(dim=-1)
    mask = y.ne(IGNORE_LABEL_ID)
    return ((pred == y) | ~mask).all(dim=1)

def training_objective(out, y, train_cfg):
    task_loss = prediction_loss(out, y, train_cfg.get('loss_type', 'cross_entropy'))
    total_loss = task_loss
    parts = {'task_loss': task_loss}
    if 'q_halt_logits' in out:
        target = exact_correct_from_output(out, y).to(out['q_halt_logits'].dtype)
        q_loss = F.binary_cross_entropy_with_logits(out['q_halt_logits'], target)
        total_loss = total_loss + float(train_cfg.get('q_halt_weight', 0.5)) * q_loss
        parts['q_halt_loss'] = q_loss
        with torch.no_grad():
            q_pred = out['q_halt_logits'] > 0
            parts['q_halt_acc'] = (q_pred == target.bool()).float().mean()
    return total_loss, parts

def _depth_forward(model, x, train_cfg):
    forward_depth = getattr(model, 'forward_depth', None)
    if forward_depth is None:
        return model(x)
    return forward_depth(
        x,
        eval_depth=train_cfg.get('eval_depth', 1),
        init_noise_std=train_cfg.get('eval_init_noise_std', 0.0),
        noise_scale=train_cfg.get('eval_noise_scale', 0.0),
    )

def _train_forward(model, x, train_cfg):
    train_depth = int(train_cfg.get('train_depth') or 1)
    actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    forward_depth = getattr(actual_model, 'forward_depth', None)
    if train_depth <= 1 or forward_depth is None:
        return model(x)
    return forward_depth(
        x,
        eval_depth=train_depth,
        init_noise_std=train_cfg.get('train_init_noise_std', 0.0),
        noise_scale=train_cfg.get('train_noise_scale', train_cfg.get('eval_noise_scale', 0.0)),
        grad_last_only=bool(train_cfg.get('train_grad_last_only', True)),
    )

def parse_eval_grid(grid):
    if not grid:
        return []
    if isinstance(grid, str):
        points = []
        for raw in grid.split(','):
            item = raw.strip().lower()
            if not item:
                continue
            if 'x' in item:
                d, b = item.split('x', 1)
            elif ':' in item:
                d, b = item.split(':', 1)
            else:
                raise ValueError(f"bad eval_grid item {raw!r}; expected DxB, e.g. 64x128")
            points.append((int(d), int(b)))
        return points
    points = []
    for item in grid:
        if isinstance(item, dict):
            points.append((int(item['depth']), int(item['breadth'])))
        else:
            d, b = item
            points.append((int(d), int(b)))
    return points

@torch.no_grad()
def evaluate_depth_breadth(model, loader, device, train_cfg, eval_depth=None, eval_breadth=None, force=False):
    eval_depth = int(eval_depth if eval_depth is not None else (train_cfg.get('eval_depth') or 1))
    eval_breadth = int(eval_breadth if eval_breadth is not None else (train_cfg.get('eval_breadth') or 1))
    if eval_depth <= 1 and eval_breadth <= 1 and not force:
        return {}
    actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if not hasattr(actual_model, 'forward_depth'):
        return {}

    model.eval()
    chunk = int(train_cfg.get('eval_breadth_chunk') or min(eval_breadth, 8))
    chunk = max(1, min(chunk, eval_breadth))
    residual_window = int(train_cfg.get('eval_residual_window') or 3)
    convergence_top_k = max(1, int(train_cfg.get('convergence_top_k') or 1))
    token_total = token_correct = exact_total = exact_correct = 0
    majority_token_correct = majority_token_total = majority_exact_correct = 0
    convergence_token_correct = convergence_token_total = convergence_exact_correct = convergence_exact_total = 0
    pass_sum = any_correct = convergence_any_correct = 0

    max_batches = train_cfg.get('eval_max_batches')
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if y.ndim == 1:
            continue
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        mask = y.ne(IGNORE_LABEL_ID)
        batch_size, seq_len = y.shape
        votes = None
        first_pred = None
        residual_chunks = []
        pred_chunks = []
        batch_pass = torch.zeros(batch_size, dtype=torch.long, device=device)
        done = 0
        while done < eval_breadth:
            n = min(chunk, eval_breadth - done)
            xb = x.repeat_interleave(n, dim=0)
            with autocast_context(device, train_cfg):
                out = actual_model.forward_depth(
                    xb,
                    eval_depth=eval_depth,
                    init_noise_std=train_cfg.get('eval_init_noise_std', 0.0),
                    noise_scale=train_cfg.get('eval_noise_scale', 0.0),
                    residual_window=residual_window,
                )
            logits = out['token_logits']
            vocab = logits.shape[-1]
            pred = logits.argmax(dim=-1).view(batch_size, n, seq_len)
            residual = out.get('residual_score')
            if first_pred is None:
                first_pred = pred[:, 0]
            if votes is None:
                votes = torch.zeros(batch_size, seq_len, vocab, dtype=torch.int32, device=device)
            vote_index = pred.transpose(1, 2)
            vote_src = torch.ones(batch_size, seq_len, n, dtype=torch.int32, device=device)
            votes.scatter_add_(2, vote_index, vote_src)
            exact = ((pred == y[:, None, :]) | ~mask[:, None, :]).all(dim=-1)
            batch_pass += exact.sum(dim=1)
            if residual is not None:
                residual_chunks.append(residual.view(batch_size, n))
                pred_chunks.append(pred)
            done += n

        assert votes is not None and first_pred is not None
        token_total += mask.sum().item()
        token_correct += ((first_pred == y) & mask).sum().item()
        exact_total += batch_size
        exact_correct += (((first_pred == y) | ~mask).all(dim=1)).sum().item()

        majority_pred = votes.argmax(dim=-1)
        majority_token_total += mask.sum().item()
        majority_token_correct += ((majority_pred == y) & mask).sum().item()
        majority_exact_correct += (((majority_pred == y) | ~mask).all(dim=1)).sum().item()
        if residual_chunks:
            all_residual = torch.cat(residual_chunks, dim=1)
            all_pred = torch.cat(pred_chunks, dim=1)
            top_k = min(convergence_top_k, all_residual.shape[1])
            best_idx = torch.argsort(all_residual, dim=1)[:, :top_k]
            best_pred = all_pred.gather(1, best_idx[:, :, None].expand(-1, -1, seq_len))
            best_exact = ((best_pred == y[:, None, :]) | ~mask[:, None, :]).all(dim=-1)
            conv_token_ok = ((best_pred == y[:, None, :]) & mask[:, None, :]).sum().item()
            convergence_token_total += mask.sum().item()
            convergence_token_correct += conv_token_ok / top_k
            convergence_exact_total += batch_size
            convergence_exact_correct += best_exact.float().mean(dim=1).sum().item()
            convergence_any_correct += best_exact.any(dim=1).sum().item()
        pass_sum += batch_pass.sum().item()
        any_correct += (batch_pass > 0).sum().item()

    if exact_total == 0:
        return {}
    return {
        'eval_depth': eval_depth,
        'eval_breadth': eval_breadth,
        'eval_depth_token_acc': token_correct / max(1, token_total),
        'eval_depth_exact_acc': exact_correct / max(1, exact_total),
        'eval_majority_token_acc': majority_token_correct / max(1, majority_token_total),
        'eval_majority_exact_acc': majority_exact_correct / max(1, exact_total),
        'eval_convergence_token_acc': convergence_token_correct / max(1, convergence_token_total),
        'eval_convergence_exact_acc': convergence_exact_correct / max(1, convergence_exact_total),
        'eval_convergence_any_correct': convergence_any_correct / max(1, convergence_exact_total),
        'eval_convergence_top_k': convergence_top_k,
        'eval_avg_pass_rate': pass_sum / max(1, exact_total * eval_breadth),
        'eval_any_correct': any_correct / max(1, exact_total),
    }

@torch.no_grad()
def evaluate_compute_frontier(model, loader, device, train_cfg):
    grid = parse_eval_grid(train_cfg.get('eval_grid'))
    if not grid:
        return {}

    frontier = []
    flat = {}
    for depth, breadth in grid:
        point_metrics = evaluate_depth_breadth(model, loader, device, train_cfg, eval_depth=depth, eval_breadth=breadth, force=True)
        if not point_metrics:
            continue
        point = {
            'depth': depth,
            'breadth': breadth,
            'nfe': depth * breadth,
            'token_acc': point_metrics.get('eval_depth_token_acc'),
            'exact_acc': point_metrics.get('eval_depth_exact_acc'),
            'majority_token_acc': point_metrics.get('eval_majority_token_acc'),
            'majority_exact_acc': point_metrics.get('eval_majority_exact_acc'),
            'convergence_token_acc': point_metrics.get('eval_convergence_token_acc'),
            'convergence_exact_acc': point_metrics.get('eval_convergence_exact_acc'),
            'convergence_any_correct': point_metrics.get('eval_convergence_any_correct'),
            'convergence_top_k': point_metrics.get('eval_convergence_top_k'),
            'avg_pass_rate': point_metrics.get('eval_avg_pass_rate'),
            'any_correct': point_metrics.get('eval_any_correct'),
        }
        frontier.append(point)
        prefix = f'frontier_d{depth}_b{breadth}'
        for key, value in point.items():
            if key not in ('depth', 'breadth'):
                flat[f'{prefix}_{key}'] = value

    if not frontier:
        return {}

    # Keep legacy eval_* columns aligned to the largest compute point.
    last = frontier[-1]
    flat.update({
        'eval_depth': last['depth'],
        'eval_breadth': last['breadth'],
        'eval_depth_token_acc': last['token_acc'],
        'eval_depth_exact_acc': last['exact_acc'],
        'eval_majority_token_acc': last['majority_token_acc'],
        'eval_majority_exact_acc': last['majority_exact_acc'],
        'eval_convergence_token_acc': last['convergence_token_acc'],
        'eval_convergence_exact_acc': last['convergence_exact_acc'],
        'eval_convergence_any_correct': last['convergence_any_correct'],
        'eval_convergence_top_k': last['convergence_top_k'],
        'eval_avg_pass_rate': last['avg_pass_rate'],
        'eval_any_correct': last['any_correct'],
        'eval_frontier': frontier,
    })
    return flat

class EMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for name, p in model.named_parameters():
            if name in backup:
                p.copy_(backup[name])

def _set_runtime_step(model, step, train_cfg):
    actual = model._orig_mod if hasattr(model, '_orig_mod') else model
    for module in actual.modules():
        if hasattr(module, 'set_runtime_step'):
            module.set_runtime_step(step, train_cfg)

def _lr_for_step(base_lr, step, total_steps, train_cfg):
    warmup = int(train_cfg.get('lr_warmup_steps') or 0)
    min_ratio = float(train_cfg.get('lr_min_ratio', 1.0))
    if warmup > 0 and step < warmup:
        return float(base_lr) * float(step + 1) / float(max(1, warmup))
    if min_ratio >= 1.0 or total_steps <= warmup:
        return float(base_lr)
    progress = float(step - warmup) / float(max(1, total_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (min_ratio + (1.0 - min_ratio) * cosine)

def _state_model(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model

def _save_checkpoint(path, model, opt, scaler, ema, global_step, epoch, last_metrics):
    tmp_path = f'{path}.tmp'
    payload = {
        'model': _state_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'global_step': int(global_step),
        'epoch': int(epoch),
        'last_metrics': last_metrics,
    }
    if ema is not None:
        payload['ema'] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
        payload['ema_decay'] = ema.decay
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

def _load_checkpoint(path, model, opt, scaler, ema, device):
    ckpt = torch.load(path, map_location=device)
    _state_model(model).load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    if ema is not None and 'ema' in ckpt:
        ema.shadow = {k: v.to(device=device) for k, v in ckpt['ema'].items()}
        ema.decay = float(ckpt.get('ema_decay', ema.decay))
    return int(ckpt.get('global_step', 0)), int(ckpt.get('epoch', 0)), ckpt.get('last_metrics')

@torch.no_grad()
def evaluate(model, loader, device, train_cfg=None):
    train_cfg = train_cfg or {}
    model.eval()
    total = correct = token_total = token_correct = exact_total = exact_correct = 0
    loss_sum = 0.0
    polar_sum = polar_count = 0
    lora_sum = lora_count = 0
    max_batches = train_cfg.get('eval_max_batches')
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_context(device, train_cfg):
            out = model(x)
            loss = prediction_loss(out, y, train_cfg.get('loss_type', 'cross_entropy'))
        if y.ndim == 1:
            pred = out['logits'].argmax(dim=-1)
            total += y.numel()
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * y.numel()
        else:
            pred = out['token_logits'].argmax(dim=-1)
            mask = y.ne(IGNORE_LABEL_ID)
            batch_token_total = mask.sum().item()
            token_total += batch_token_total
            token_correct += ((pred == y) & mask).sum().item()
            exact_total += y.shape[0]
            exact_correct += (((pred == y) | ~mask).all(dim=1)).sum().item()
            loss_sum += loss.item() * max(1, batch_token_total)
        if 'polar_usage' in out:
            polar_sum += out['polar_usage'].item(); polar_count += 1
        if 'lora_delta_norm' in out:
            lora_sum += out['lora_delta_norm'].item(); lora_count += 1
    denom = total if total else token_total
    metrics = {'val_loss': loss_sum / max(1, denom)}
    if total:
        metrics['val_acc'] = correct / total
    if token_total:
        metrics['val_token_acc'] = token_correct / token_total
        metrics['val_exact_acc'] = exact_correct / max(1, exact_total)
    if polar_count:
        metrics['polar_usage'] = polar_sum / polar_count
    if lora_count:
        metrics['lora_delta_norm'] = lora_sum / lora_count
    frontier_metrics = evaluate_compute_frontier(model, loader, device, train_cfg)
    metrics.update(frontier_metrics or evaluate_depth_breadth(model, loader, device, train_cfg))
    return metrics

def train_from_config(model_cfg, task_cfg, train_cfg):
    validate_config(model_cfg, task_cfg)
    set_seed(train_cfg.get('seed', 0))
    device = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get('allow_tf32', True))
        torch.backends.cudnn.allow_tf32 = bool(train_cfg.get('allow_tf32', True))
        torch.set_float32_matmul_precision(train_cfg.get('float32_matmul_precision', 'high'))
    train_loader, val_loader = make_dataloaders(
        task_cfg,
        train_cfg.get('batch_size', 128),
        train_cfg.get('num_workers', 0),
        train_cfg.get('pin_memory', device.startswith('cuda')),
        train_cfg.get('persistent_workers', train_cfg.get('num_workers', 0) > 0),
    )
    model = build_model(model_cfg).to(device)
    if train_cfg.get('compile', False):
        model = torch.compile(model, mode=train_cfg.get('compile_mode', 'default'))
    n_params = count_parameters(model)
    base_lr = train_cfg.get('lr', 3e-4)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_cfg.get('weight_decay', 0.01),
        betas=(train_cfg.get('beta1', 0.9), train_cfg.get('beta2', 0.95)),
    )
    ema = EMA(model, train_cfg.get('ema_rate', 0.999)) if train_cfg.get('ema', False) else None
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda') and train_cfg.get('amp', True) and train_cfg.get('amp_dtype', 'bfloat16') == 'float16')
    out_dir = train_cfg.get('out_dir', f'runs/{model_cfg.model_type}')
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'model': asdict(model_cfg), 'task': asdict(task_cfg), 'train': train_cfg, 'n_params': n_params}, open(os.path.join(out_dir, 'config_resolved.json'), 'w'), indent=2)
    metrics_path = os.path.join(out_dir, 'metrics.jsonl')
    checkpoint_path = os.path.join(out_dir, 'checkpoint.pt')
    print(f'model={model_cfg.model_type} params={n_params:,} device={device}')
    global_step = 0
    max_steps = train_cfg.get('max_steps')
    total_steps = int(max_steps or (train_cfg.get('epochs', 10) * max(1, len(train_loader))))
    eval_interval_steps = train_cfg.get('eval_interval_steps')
    last_metrics = None
    start_epoch = 0
    if train_cfg.get('resume', True) and os.path.exists(checkpoint_path):
        global_step, start_epoch, last_metrics = _load_checkpoint(checkpoint_path, model, opt, scaler, ema, device)
        print(f'resumed checkpoint from {checkpoint_path} at step={global_step} epoch={start_epoch}')
    checkpoint_interval_steps = train_cfg.get('checkpoint_interval_steps', eval_interval_steps)
    for epoch in range(start_epoch, train_cfg.get('epochs', 10)):
        model.train(); pbar = tqdm(train_loader, desc=f'{model_cfg.model_type} epoch {epoch}')
        for x, y in pbar:
            if max_steps is not None and global_step >= max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _set_runtime_step(model, global_step, train_cfg)
            lr_this_step = _lr_for_step(base_lr, global_step, total_steps, train_cfg)
            for group in opt.param_groups:
                group['lr'] = lr_this_step
            with autocast_context(device, train_cfg):
                out = _train_forward(model, x, train_cfg)
                loss, loss_parts = training_objective(out, y, train_cfg)
                task_loss = loss_parts['task_loss']
                div = torch.tensor(0.0, device=device)
                if model_cfg.model_type.startswith(('lg_prm', 'lgp')) and model_cfg.use_diversity_loss and 'proposals' in out:
                    div = diversity_loss(out['proposals']); loss = loss + train_cfg.get('diversity_weight', 0.0) * div
                if 'polar_usage' in out:
                    loss = loss + train_cfg.get('polar_weight', 0.0) * out['polar_usage']
                if 'lora_delta_norm' in out:
                    loss = loss + train_cfg.get('lora_delta_weight', 0.0) * out['lora_delta_norm']
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get('grad_clip', 1.0))
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            global_step += 1
            postfix = {'loss': f'{loss.item():.3f}', 'task': f'{task_loss.item():.3f}', 'lr': f'{lr_this_step:.1e}', 'div': f'{div.item():.3f}'}
            if 'q_halt_loss' in loss_parts:
                postfix['q'] = f"{loss_parts['q_halt_loss'].item():.3f}"
            pbar.set_postfix(postfix)
            if eval_interval_steps and global_step % eval_interval_steps == 0:
                backup = ema.apply_to(model) if ema is not None else None
                metrics = evaluate(model, val_loader, device, train_cfg)
                if ema is not None:
                    ema.restore(model, backup)
                metrics.update({'epoch': epoch, 'step': global_step, 'n_params': n_params, 'model_type': model_cfg.model_type})
                print(metrics)
                with open(metrics_path, 'a') as f: f.write(json.dumps(metrics) + '\n')
                last_metrics = metrics
                _save_checkpoint(checkpoint_path, model, opt, scaler, ema, global_step, epoch, last_metrics)
                model.train()
            elif checkpoint_interval_steps and global_step % int(checkpoint_interval_steps) == 0:
                _save_checkpoint(checkpoint_path, model, opt, scaler, ema, global_step, epoch, last_metrics)
        if max_steps is not None and global_step >= max_steps:
            break
        if not eval_interval_steps:
            backup = ema.apply_to(model) if ema is not None else None
            metrics = evaluate(model, val_loader, device, train_cfg)
            if ema is not None:
                ema.restore(model, backup)
            metrics.update({'epoch': epoch, 'step': global_step, 'n_params': n_params, 'model_type': model_cfg.model_type})
            print(metrics)
            with open(metrics_path, 'a') as f: f.write(json.dumps(metrics) + '\n')
            last_metrics = metrics
            _save_checkpoint(checkpoint_path, model, opt, scaler, ema, global_step, epoch, last_metrics)
    backup = ema.apply_to(model) if ema is not None else None
    metrics = evaluate(model, val_loader, device, train_cfg)
    if ema is not None:
        ema.restore(model, backup)
    metrics.update({'epoch': epoch, 'step': global_step, 'n_params': n_params, 'model_type': model_cfg.model_type})
    if last_metrics is None or last_metrics.get('step') != global_step:
        print(metrics)
        with open(metrics_path, 'a') as f: f.write(json.dumps(metrics) + '\n')
    _save_checkpoint(checkpoint_path, model, opt, scaler, ema, global_step, epoch, metrics)
    model_to_save = _state_model(model)
    torch.save(model_to_save.state_dict(), os.path.join(out_dir, 'model.pt'))
    return metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True)
    args = ap.parse_args(); train_from_config(*load_config(args.config))
