import argparse
import json
from pathlib import Path
from sglm.manifest import analyze_manifest

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    report = analyze_manifest(a.manifest)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'groups'}, indent=2))
