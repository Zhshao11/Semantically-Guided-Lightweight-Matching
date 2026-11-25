"""Metadata-only finalization. No image warp, affine fit or model execution."""
import argparse
import json
from sglm.metadata_finalize import freeze_metadata

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--audit',required=True)
    ap.add_argument('--control-points',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    print(json.dumps(freeze_metadata(a.manifest,a.audit,a.control_points,a.output),indent=2))
