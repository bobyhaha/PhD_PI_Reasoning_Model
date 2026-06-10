import random
from dataclasses import dataclass
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class TaskConfig:
    task_name: str = 'mod_expr'
    seq_len: int = 31
    modulus: int = 17
    n_train: int = 20000
    n_val: int = 2000
    vocab_size: int = 128
    seed: int = 0
    dataset_path: str = ''
    puzzle_set: str = ''

class ReasoningDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        n = cfg.n_train if split == 'train' else cfg.n_val
        rng = random.Random(cfg.seed + (0 if split == 'train' else 10000))
        self.samples = [self._make_sample(rng) for _ in range(n)]
    @property
    def pad_id(self): return self.cfg.modulus + 3
    @property
    def plus_id(self): return self.cfg.modulus
    @property
    def minus_id(self): return self.cfg.modulus + 1
    @property
    def times_id(self): return self.cfg.modulus + 2
    def _make_mod_expr(self, rng):
        M = self.cfg.modulus
        n_terms = rng.randint(3, (self.cfg.seq_len + 1) // 2)
        nums = [rng.randrange(M) for _ in range(n_terms)]
        ops = [rng.choice([self.plus_id, self.minus_id, self.times_id]) for _ in range(n_terms - 1)]
        val = nums[0]
        toks = [nums[0]]
        for op, num in zip(ops, nums[1:]):
            toks += [op, num]
            if op == self.plus_id: val = (val + num) % M
            elif op == self.minus_id: val = (val - num) % M
            else: val = (val * num) % M
        toks = toks[:self.cfg.seq_len] + [self.pad_id] * max(0, self.cfg.seq_len - len(toks))
        return toks, val
    def _make_copy_last(self, rng):
        M = self.cfg.modulus
        length = rng.randint(2, self.cfg.seq_len)
        toks = [rng.randrange(M) for _ in range(length)]
        y = toks[-1]
        toks += [self.pad_id] * (self.cfg.seq_len - len(toks))
        return toks, y
    def _make_parity(self, rng):
        M = self.cfg.modulus
        length = rng.randint(2, self.cfg.seq_len)
        toks = [rng.randrange(M) for _ in range(length)]
        y = sum(toks) % 2
        toks += [self.pad_id] * (self.cfg.seq_len - len(toks))
        return toks, y
    def _make_sample(self, rng):
        if self.cfg.task_name == 'mod_expr': return self._make_mod_expr(rng)
        if self.cfg.task_name == 'copy_last': return self._make_copy_last(rng)
        if self.cfg.task_name == 'parity': return self._make_parity(rng)
        raise ValueError(self.cfg.task_name)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class PuzzleNpyDataset(Dataset):
    def __init__(self, cfg, split='train'):
        if not cfg.dataset_path:
            raise ValueError('dataset_path is required for task_name=puzzle_npy')
        root = self._split_root(cfg.dataset_path, split)
        meta_path = os.path.join(root, 'dataset.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        set_name = cfg.puzzle_set or meta.get('sets', ['all'])[0]
        self.inputs = np.load(os.path.join(root, f'{set_name}__inputs.npy'), mmap_mode='r')
        self.labels = np.load(os.path.join(root, f'{set_name}__labels.npy'), mmap_mode='r')
        self.ignore_label_id = meta.get('ignore_label_id')
        self.limit = cfg.n_train if split == 'train' else cfg.n_val
        self.limit = min(self.limit, len(self.inputs))
    def __len__(self):
        return self.limit
    def __getitem__(self, i):
        x = torch.tensor(self.inputs[i], dtype=torch.long)
        y = torch.tensor(self.labels[i].copy(), dtype=torch.long)
        if self.ignore_label_id is not None:
            y[y == int(self.ignore_label_id)] = -100
        return x, y

    @staticmethod
    def _split_root(dataset_path, split):
        candidates = [split]
        if split == 'val':
            candidates.append('test')
        for name in candidates:
            root = os.path.join(dataset_path, name)
            if os.path.isdir(root):
                return root
        raise FileNotFoundError(os.path.join(dataset_path, split))

def make_dataloaders(cfg, batch_size, num_workers=0, pin_memory=False, persistent_workers=False):
    dataset_cls = PuzzleNpyDataset if cfg.task_name == 'puzzle_npy' else ReasoningDataset
    persistent = persistent_workers and num_workers > 0
    train = DataLoader(
        dataset_cls(cfg, 'train'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val = DataLoader(
        dataset_cls(cfg, 'val'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train, val
