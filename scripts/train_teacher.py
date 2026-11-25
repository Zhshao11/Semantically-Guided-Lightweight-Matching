"""Fine-tune a SegFormer-B2 teacher on externally labeled remote-sensing data.
Manifest: JSON list of {image, label}; label pixels 0..6, ignored pixels 255.
All paths are relative to --data-root. Training and validation manifests are separate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class SegmentationData(Dataset):
    def __init__(self, root, manifest, processor):
        self.root, self.items, self.processor = Path(root), json.loads(Path(manifest).read_text()), processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(self.root / item['image']).convert('RGB')
        label = np.asarray(Image.open(self.root / item['label'])).copy()
        if not np.isin(label, list(range(7)) + [255]).all():
            raise ValueError('Map labels explicitly to 0..6 and ignore=255 before training.')
        return self.processor(images=image, segmentation_maps=label, return_tensors='pt').data


def main():
    ap = argparse.ArgumentParser()
    for name in ('data-root', 'train-manifest', 'val-manifest', 'initial-model', 'class-map', 'output'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    torch.manual_seed(20261125)
    classes = json.loads(Path(a.class_map).read_text())
    if set(classes) != set(map(str, range(7))):
        raise ValueError('class-map must map keys 0..6 to class names')
    processor = AutoImageProcessor.from_pretrained(a.initial_model, local_files_only=True, do_reduce_labels=False)
    model = SegformerForSemanticSegmentation.from_pretrained(a.initial_model, local_files_only=True,
              num_labels=7, id2label={int(k): v for k, v in classes.items()},
              label2id={v: int(k) for k, v in classes.items()}, ignore_mismatched_sizes=True).to(a.device)
    if list(model.config.hidden_sizes) != [64, 128, 320, 512] or list(model.config.depths) != [3, 4, 6, 3]:
        raise ValueError('Expected SegFormer-B2 backbone configuration.')
    model.config.semantic_loss_ignore_index = 255
    train = SegmentationData(a.data_root, a.train_manifest, processor)
    val = SegmentationData(a.data_root, a.val_manifest, processor)
    tr_ids = {str((Path(a.data_root) / x['image']).resolve()) for x in train.items}
    va_ids = {str((Path(a.data_root) / x['image']).resolve()) for x in val.items}
    if tr_ids & va_ids:
        raise ValueError('Teacher train and validation image paths overlap')
    loaders = [DataLoader(ds, batch_size=a.batch_size, shuffle=(i == 0)) for i, ds in enumerate((train, val))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=.01)
    best = -1
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        for batch in loaders[0]:
            batch = {k: v.squeeze(1).to(a.device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        model.eval()
        confusion = torch.zeros(7, 7, device=a.device)
        with torch.no_grad():
            for batch in loaders[1]:
                batch = {k: v.squeeze(1).to(a.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                pred = F.interpolate(model(**batch).logits, size=labels.shape[-2:], mode='bilinear', align_corners=False).argmax(1)
                valid = labels != 255
                confusion += torch.bincount((labels[valid] * 7 + pred[valid]).long(), minlength=49).reshape(7, 7)
        union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
        if not (union > 0).any():
            raise ValueError('No valid validation labels')
        miou = (confusion.diag()[union > 0] / union[union > 0]).mean().item()
        print(json.dumps(dict(epoch=epoch, miou=miou)), flush=True)
        if miou > best:
            best = miou
            model.save_pretrained(out)
            processor.save_pretrained(out)
            (out / 'training_metadata.json').write_text(json.dumps(dict(config=vars(a), best_val_miou=best), indent=2))


if __name__ == '__main__':
    main()
