"""Official full LoveDA Train ZIP, resumable ranges; fixed64 new Val members."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request
import zipfile
from fetch_loveda_val_subset import RangeFile


class RetryingRangeFile(RangeFile):
    def read(self,n=-1):
        for attempt in range(5):
            try:return super().read(n)
            except Exception:
                if attempt==4:raise
                self.cache=b'';time.sleep(min(10,2**attempt))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='runs/t06_teacher_data');ap.add_argument('--train-md5',required=True)
    ap.add_argument('--old8-manifest',default='runs/loveda_val_t05_v1/manifest.json')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    size=4021669263;chunk=64*1024**2;url='https://zenodo.org/api/records/5706578/files/Train.zip/content'
    def download(i):
        start=i*chunk;end=min(size,start+chunk);target=out/f'Train.chunk{i:03d}'
        if target.exists() and target.stat().st_size==end-start:return target
        for attempt in range(4):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end-1}'})
                with urllib.request.urlopen(req,timeout=120) as response:
                    if response.status!=206 or not response.headers['Content-Range'].startswith(f'bytes {start}-'):raise IOError('Range not honored')
                    with target.open('wb') as stream:
                        while True:
                            data=response.read(1024**2)
                            if not data:break
                            stream.write(data)
                if target.stat().st_size!=end-start:raise IOError('Incomplete chunk')
                return target
            except Exception:
                if attempt==3:raise
                time.sleep(min(10,2**attempt))
    archive=out/'Train.zip'
    if not archive.exists():
        pieces={};count=(size+chunk-1)//chunk
        with ThreadPoolExecutor(24) as pool:
            for future in as_completed({pool.submit(download,i):i for i in range(count)}):
                p=future.result();pieces[int(p.name[-3:])]=p;print('train chunk completed',len(pieces),count,flush=True)
                (out/'download_status.json').write_text(json.dumps(dict(train_chunks_complete=len(pieces),train_chunks_total=count)))
        with archive.open('wb') as stream:
            for i in range(count):
                with pieces[i].open('rb') as part:
                    while data:=part.read(1024**2):stream.write(data)
    md5=hashlib.md5()
    with archive.open('rb') as stream:
        while data:=stream.read(4*1024**2):md5.update(data)
    if md5.hexdigest()!=a.train_md5:raise ValueError('Official complete Train ZIP MD5 mismatch')
    train=[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if '/images_png/' not in name or not name.endswith('.png'):continue
            label=name.replace('/images_png/','/masks_png/');domain='Rural' if '/Rural/' in name else 'Urban'
            row=dict(pair_id=domain+'/'+Path(name).stem,domain=domain,split='Train',files={})
            for kind,member in [('image',name),('label',label)]:
                data=z.read(member);p=out/'Train'/domain/kind/Path(member).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                row['files'][kind]=dict(path=str(p.relative_to(out)),sha256=hashlib.sha256(data).hexdigest(),archive_member=member)
            train.append(row)
    assert len(train)==2522
    # New independent64 excludes old8, whose hashes/data remain available separately.
    old=json.loads(Path(a.old8_manifest).read_text())['samples'];oldids={r['pair_id'] for r in old}
    val=[];remote=RetryingRangeFile('https://zenodo.org/api/records/5706578/files/Val.zip/content',2425958254)
    with zipfile.ZipFile(remote) as z:
        names=z.namelist();rng=random.Random(20261126)
        for domain in ['Rural','Urban']:
            candidates=sorted(n for n in names if f'/{domain}/images_png/' in '/'+n and n.endswith('.png') and domain+'/'+Path(n).stem not in oldids)
            for name in sorted(rng.sample(candidates,32)):
                row=dict(pair_id=domain+'/'+Path(name).stem,domain=domain,split='Val',files={})
                for kind,member in [('image',name),('label',name.replace('/images_png/','/masks_png/'))]:
                    import zlib
                    p=out/'Val64'/domain/kind/Path(member).name
                    existing=p.read_bytes() if p.exists() else b''
                    data=existing if existing and zlib.crc32(existing)==z.getinfo(member).CRC else z.read(member)
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                    row['files'][kind]=dict(path=str(p.relative_to(out)),sha256=hashlib.sha256(data).hexdigest(),archive_member=member,CRC_verified=True)
                val.append(row);print('validation fetched',len(val),64,row['pair_id'],flush=True)
    report=dict(source='https://zenodo.org/records/5706578',license='CC-BY-4.0',train_zip_md5=md5.hexdigest(),train_zip_complete_MD5_verified=True,
        train=train,val64=val,old8_manifest='runs/loveda_val_t05_v1/manifest.json',val_excludes_old8=True,test_used=False,seed=20261126)
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print('Dataset prepared2522 train+64 new val',flush=True)


if __name__=='__main__':main()
