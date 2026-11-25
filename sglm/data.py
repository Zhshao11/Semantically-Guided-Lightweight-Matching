"""OSdataset loader: original split, deterministic synthetic affine GT, no test tuning."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def apply_transform(points, matrix):
    return points @ matrix[..., :2].T + matrix[..., 2]


def random_affine(rng, width, height):
    angle = rng.uniform(-35, 35) * np.pi / 180
    scale = rng.uniform(.7, 1.3)
    linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    translation = rng.uniform(-.1, .1, 2) * [width, height]
    return np.column_stack([linear, center + translation - linear @ center]).astype(np.float32)


class OSPairs(Dataset):
    def __init__(self, root, split='train', seed=20261125, affine=True, limit=None, semantic_cache=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.folder = Path(root) / 'Patch-level Subset/OSdataset/256' / split
        if not self.folder.is_dir():
            raise FileNotFoundError(self.folder)
        self.paths = sorted(self.folder.glob('opt*.png'), key=lambda p: int(p.stem[3:]))
        if not self.paths:
            raise ValueError(f'No pairs in {self.folder}')
        for p in self.paths:
            if not p.with_name('sar' + p.name[3:]).is_file():
                raise FileNotFoundError(p)
        if limit is not None:
            if limit <= 0:
                raise ValueError('limit must be positive')
            self.paths = self.paths[:limit]
        self.split, self.seed, self.affine, self.epoch = split, seed, affine, 0
        self.cache = Path(semantic_cache) if semantic_cache else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        p = self.paths[index]
        optical = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        sar = cv2.imread(str(p.with_name('sar' + p.name[3:])), cv2.IMREAD_GRAYSCALE)
        h, w = sar.shape
        # Per-pair hash avoids dependence on filesystem ordering or workers.
        epoch = self.epoch if self.split == 'train' else 0
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}/{self.split}/{p.stem}/{epoch}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        matrix = random_affine(rng, w, h) if self.affine else np.eye(2, 3, dtype=np.float32)
        # OpenCV uses a source->destination map unless WARP_INVERSE_MAP is specified.
        sar = cv2.warpAffine(sar, matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.ones((h, w), np.float32), matrix, (w, h), flags=cv2.INTER_LINEAR) > .999
        item = dict(optical=torch.from_numpy(optical.copy()).permute(2, 0, 1).float() / 255,
                    sar=torch.from_numpy(sar.copy())[None].float() / 255,
                    optical_mask=torch.ones(h, w, dtype=torch.bool), sar_mask=torch.from_numpy(mask),
                    transform=torch.from_numpy(matrix), id=p.stem)
        if self.cache:
            cache_file = self.cache / self.split / (p.stem + '.npz')
            with np.load(cache_file, allow_pickle=False) as cache:
                t = cache['probabilities'].astype(np.float32)
                m = cache['mask'].astype(bool)
            if t.shape != (7, h // 8, w // 8) or m.shape != (h // 8, w // 8):
                raise ValueError(f'Invalid semantic cache shape: {cache_file}')
            item.update(teacher=torch.from_numpy(t), teacher_mask=torch.from_numpy(m))
        return item
