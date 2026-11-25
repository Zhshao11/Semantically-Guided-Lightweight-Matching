"""Cache seven-class optical soft targets using a user-supplied trained teacher."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--teacher', required=True, help='Local save_pretrained directory, trained for seven land-cover classes')
    ap.add_argument('--output', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--min-confidence', type=float, default=.6)
    a = ap.parse_args()
    processor = AutoImageProcessor.from_pretrained(a.teacher, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(a.teacher, local_files_only=True).eval().to(a.device)
    if model.config.num_labels != 7:
        raise ValueError('Teacher must have seven classes with a documented id2label mapping.')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metadata.json').write_text(json.dumps(dict(teacher=str(Path(a.teacher).resolve()),
        id2label=model.config.id2label, min_confidence=a.min_confidence,
        note='Probabilities resized to H/8,W/8 then renormalized. Mask is teacher confidence, not geometric GT.'), indent=2))
    with torch.no_grad():
        for split in a.splits:
            folder = Path(a.root) / 'Patch-level Subset/OSdataset/256' / split
            (out / split).mkdir(exist_ok=True)
            for p in sorted(folder.glob('opt*.png')):
                image = Image.open(p).convert('RGB')
                data = processor(images=image, return_tensors='pt').to(a.device)
                logits = model(**data).logits
                prob = F.interpolate(logits.softmax(1), size=(image.height // 8, image.width // 8),
                                     mode='bilinear', align_corners=False)[0]
                prob = prob / prob.sum(0, keepdim=True).clamp_min(1e-8)
                mask = prob.max(0).values >= a.min_confidence
                np.savez_compressed(out / split / (p.stem + '.npz'),
                                    probabilities=prob.cpu().numpy().astype(np.float16), mask=mask.cpu().numpy())


if __name__ == '__main__':
    main()
