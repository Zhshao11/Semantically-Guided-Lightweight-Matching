"""Exhaustive split, pixel, archive, provenance and control-point audit."""
import argparse
from sglm.audit import audit_dataset

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', help='OSDataset2.0 root; otherwise SGLM_DATA_ROOT')
    ap.add_argument('--archive', help='Local original ZIP for extracted size/CRC comparison')
    ap.add_argument('--config', help='Portable data/geometry JSON; defaults to configs/data_geometry_v1.json')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    report = audit_dataset(args.root, args.output, args.archive, args.config)
    raise SystemExit(1 if report['errors'] else 0)
