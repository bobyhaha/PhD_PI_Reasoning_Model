import argparse, json, os, random
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

def prediction_loss(out, y):
    if y.ndim == 1:
        return F.cross_entropy(out['logits'], y)
    if 'token_logits' not in out:
        raise ValueError('sequence-label training requires model outputs with token_logits')
    return F.cross_entropy(out['token_logits'].reshape(-1, out['token_logits'].shape[-1]), y.reshape(-1), ignore_index=IGNORE_LABEL_ID)

@torch.no_grad()
def evaluate(model, loader, device, train_cfg=None):
    train_cfg = train_cfg or {}
    model.eval()
    total = correct = token_total = token_correct = exact_total = exact_correct = 0
    loss_sum = gate_sum = hard_gate_sum = entropy_sum = 0.0
    gate_count = hard_gate_count = entropy_count = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_context(device, train_cfg):
            out = model(x)
            loss = prediction_loss(out, y)
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
        if 'gate_probs' in out:
            gate_sum += out['gate_probs'].mean().item(); gate_count += 1
        if 'gate_hard' in out:
            hard_gate_sum += out['gate_hard'].mean().item(); hard_gate_count += 1
        if 'library_entropy' in out:
            entropy_sum += out['library_entropy'].item(); entropy_count += 1
    denom = total if total else token_total
    metrics = {'val_loss': loss_sum / max(1, denom), 'mean_gate': gate_sum / max(1, gate_count)}
    if total:
        metrics['val_acc'] = correct / total
    if token_total:
        metrics['val_token_acc'] = token_correct / token_total
        metrics['val_exact_acc'] = exact_correct / max(1, exact_total)
    if hard_gate_count:
        metrics['mean_hard_gate'] = hard_gate_sum / hard_gate_count
    if entropy_count:
        metrics['library_entropy'] = entropy_sum / entropy_count
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
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.get('lr', 3e-4), weight_decay=train_cfg.get('weight_decay', 0.01))
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda') and train_cfg.get('amp', True) and train_cfg.get('amp_dtype', 'bfloat16') == 'float16')
    out_dir = train_cfg.get('out_dir', f'runs/{model_cfg.model_type}')
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'model': asdict(model_cfg), 'task': asdict(task_cfg), 'train': train_cfg, 'n_params': n_params}, open(os.path.join(out_dir, 'config_resolved.json'), 'w'), indent=2)
    metrics_path = os.path.join(out_dir, 'metrics.jsonl')
    print(f'model={model_cfg.model_type} params={n_params:,} device={device}')
    for epoch in range(train_cfg.get('epochs', 10)):
        model.train(); pbar = tqdm(train_loader, desc=f'{model_cfg.model_type} epoch {epoch}')
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_context(device, train_cfg):
                out = model(x)
                task_loss = prediction_loss(out, y)
                loss = task_loss
                gate_cost = torch.tensor(0.0, device=device)
                if 'gate_probs' in out:
                    gate_cost = out['gate_probs'].mean(); loss = loss + train_cfg.get('retrieval_cost', 0.0) * gate_cost
                div = torch.tensor(0.0, device=device)
                if model_cfg.model_type == 'lg_prm' and model_cfg.use_diversity_loss and 'proposals' in out:
                    div = diversity_loss(out['proposals']); loss = loss + train_cfg.get('diversity_weight', 0.0) * div
                if 'library_entropy' in out:
                    loss = loss - train_cfg.get('entropy_weight', 0.0) * out['library_entropy']
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get('grad_clip', 1.0))
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'task': f'{task_loss.item():.3f}', 'gate': f'{gate_cost.item():.3f}', 'div': f'{div.item():.3f}'})
        metrics = evaluate(model, val_loader, device, train_cfg)
        metrics.update({'epoch': epoch, 'n_params': n_params, 'model_type': model_cfg.model_type})
        print(metrics)
        with open(metrics_path, 'a') as f: f.write(json.dumps(metrics) + '\n')
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(model_to_save.state_dict(), os.path.join(out_dir, 'model.pt'))
    return metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True)
    args = ap.parse_args(); train_from_config(*load_config(args.config))
