"""Read-only exports and fail-closed benchmark device checks."""
import csv
import io
import json
from pathlib import Path


def complete_jsonl(path):
    path=Path(path)
    if not path.exists():return []
    lines=path.read_text().splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
            # A running writer may not yet have completed the trailing record.
    return rows


def write_csv(path,rows):
    rows=list(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def flatten_diagnostic(record):
    row=dict(pair_id=record['pair_id'])
    for key in ('group','variant_index','GT_used_only_posthoc'):
        if key in record:row[key]=record[key]
    def visit(prefix,value):
        if isinstance(value,dict):
            for key,child in value.items():visit(f'{prefix}.{key}' if prefix else key,child)
        elif not isinstance(value,(list,tuple)):row[prefix]=value
    for key in ('coarse','local_window','refinement','evaluation'):
        if key in record:visit(key,record[key])
    if 'geometry' in record:
        for key,value in record['geometry'].items():
            if key=='estimation':
                value={k:v for k,v in value.items() if not isinstance(v,(list,tuple,dict))}
            visit('geometry.'+key,value)
    return row


def require_exclusive_gpu(gpu_uuid,process_csv,own_pid,known_uuids):
    """Reject unknown device, malformed inventory, and any other CUDA process.

    The caller must query successfully; never substitute empty output on errors.
    All processes count, including inference jobs unrelated to this project.
    """
    if not gpu_uuid.startswith('GPU-') or gpu_uuid not in known_uuids:
        raise RuntimeError('An explicit, known physical GPU UUID is required')
    conflicts=[]
    for row in csv.reader(io.StringIO(process_csv)):
        if not row:continue
        if len(row)!=3:raise RuntimeError('Cannot parse GPU process inventory')
        uuid,pid,name=[x.strip() for x in row]
        if not pid.isdigit() or not uuid.startswith('GPU-'):raise RuntimeError('Invalid GPU process inventory')
        if uuid==gpu_uuid and int(pid)!=own_pid:conflicts.append(dict(pid=int(pid),name=name))
    if conflicts:raise RuntimeError(f'Formal timing requires an exclusive GPU; conflicts={conflicts}')
    return True


def fixed_timing_plan(plan,config,split):
    """Adapt saved T02 or T03 validation plans without resampling augmentation."""
    rows=[s for s in plan['samples'] if s['split']==split] if 'samples' in plan else plan[split]
    if not rows or any('A_forward' not in row for row in rows):
        raise ValueError('Timing requires saved fixed transforms; online training plan is not a timing dataset')
    return dict(config=config,samples=[dict(row,split=split) for row in rows])
