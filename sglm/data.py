"""Original OSdataset 256/512 splits with explicit geometry and optional audit manifest."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import load_config, resolve_data_root
from .geometry import affine_from_parameters, transform_points, warp_image


def apply_transform(points, matrix):
    """Compatibility entry point for a single source->destination affine."""
    return transform_points(points, matrix)


def random_affine(rng, width, height, augmentation=None):
    if augmentation is None:
        augmentation = load_config()['augmentation']
    angle = rng.uniform(*augmentation['rotation_deg'])
    scale = rng.uniform(*augmentation['scale'])
    translation = rng.uniform(*augmentation['translation_fraction'], size=2) * [width, height]
    return affine_from_parameters(width, height, angle_deg=angle, scale=(scale, scale),
                                  translation=translation).astype(np.float32)


def _read_patch(path, optical):
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f'Could not decode {path}')
    if array.dtype != np.uint8:
        raise ValueError(f'Expected uint8 patch, got {array.dtype}: {path}')
    if optical:
        if array.ndim == 2:
            return np.repeat(array[..., None], 3, axis=2)
        if array.ndim == 3 and array.shape[-1] == 3:
            return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif array.ndim == 2:
        return array
    raise ValueError(f'Unexpected patch channel shape {array.shape}: {path}')


class OSPairs(Dataset):
    def __init__(self, root=None, split='train', seed=None, affine=True, limit=None, semantic_cache=None,
                 resolution=256, manifest=None, config_path=None):
        self.config = load_config(config_path)
        self.root = resolve_data_root(root)
        if split not in ('train', 'val', 'test') or resolution not in (256, 512):
            raise ValueError('Expected train/val/test and resolution 256/512')
        if limit is not None and limit <= 0:
            raise ValueError('limit must be positive')
        relative = self.config['dataset']['builtin_path'].format(resolution=resolution, split=split)
        self.folder = self.root / relative
        if not self.folder.is_dir():
            raise FileNotFoundError(self.folder)
        self.entries = []
        if manifest:
            with Path(manifest).open() as stream:
                seen = set()
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get('schema') != self.config['dataset']['manifest_schema']:
                        raise ValueError(f'Unknown manifest schema at line {line_number}')
                    if record['dataset'] != 'OSdataset' or record['resolution'] != resolution or record['split'] != split:
                        continue
                    if not record['usable'] or record['pair_id'] in seen:
                        raise ValueError(f'Unusable/duplicate manifest pair: {record["pair_id"]}')
                    seen.add(record['pair_id'])
                    paths = []
                    for modality in ('optical', 'sar'):
                        path = self.root / record[modality]['path']
                        if path.parent != self.folder or not path.is_file():
                            raise ValueError(f'Manifest path outside selected split or missing: {path}')
                        paths.append(path)
                    self.entries.append((paths[0], paths[1], record['pair_id']))
            self.entries.sort(key=lambda x: int(x[0].stem[3:]))
        else:
            paths = sorted(self.folder.glob('opt*.png'), key=lambda p: int(p.stem[3:]))
            optical_ids = {p.stem[3:] for p in paths}
            sar_ids = {p.stem[3:] for p in self.folder.glob('sar*.png')}
            if optical_ids != sar_ids:
                raise ValueError(f'Unpaired optical/SAR files in {self.folder}')
            self.entries = [(p, p.with_name('sar' + p.name[3:]), f'osdataset/{resolution}/{split}/{p.stem[3:]}') for p in paths]
        if not self.entries:
            raise ValueError(f'No manifest/directory pairs in {self.folder}')
        if limit is not None:
            self.entries = self.entries[:limit]
        self.paths = [x[0] for x in self.entries]  # Backwards compatible inspection.
        self.split, self.seed, self.affine, self.epoch = split, self.config['seed'] if seed is None else seed, affine, 0
        self.resolution = resolution
        self.cache = Path(semantic_cache) if semantic_cache else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        p, sar_path, pair_id = self.entries[index]
        optical, sar = _read_patch(p, True), _read_patch(sar_path, False)
        h, w = sar.shape
        if optical.shape[:2] != sar.shape or (h, w) != (self.resolution, self.resolution):
            raise ValueError(f'Unexpected image dimensions for {pair_id}')
        epoch = self.epoch if self.split == 'train' else 0
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}/{self.split}/{p.stem}/{epoch}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        matrix = random_affine(rng, w, h, self.config['augmentation']) if self.affine else np.eye(2, 3, dtype=np.float32)
        sar, mask = warp_image(sar, matrix, mask_min_weight=self.config['augmentation']['mask_min_weight'])
        item = dict(optical=torch.from_numpy(optical.copy()).permute(2, 0, 1).float() / 255,
                    sar=torch.from_numpy(sar.copy())[None].float() / 255,
                    optical_mask=torch.ones(h, w, dtype=torch.bool), sar_mask=torch.from_numpy(mask),
                    transform=torch.from_numpy(matrix), id=p.stem, pair_id=pair_id)
        if self.cache:
            cache_file = self.cache / self.split / (p.stem + '.npz')
            with np.load(cache_file, allow_pickle=False) as cache:
                t = cache['probabilities'].astype(np.float32)
                m = cache['mask'].astype(bool)
            if t.shape != (7, h // 8, w // 8) or m.shape != (h // 8, w // 8):
                raise ValueError(f'Invalid semantic cache shape: {cache_file}')
            item.update(teacher=torch.from_numpy(t), teacher_mask=torch.from_numpy(m))
        return item
