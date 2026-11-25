"""Fetch a fixed, labeled LoveDA validation subset using official ZIP byte ranges."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request
import zipfile


class RangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size,self.position=url,size,0
        self.cache_start,self.cache=0,b''
        self.requests,self.transferred=0,0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size)
        if n<=0:return b''
        start=self.position;end=start+n
        if not (self.cache_start<=start and end<=self.cache_start+len(self.cache)):
            fetch_end=min(self.size,start+max(n,65536))
            request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{fetch_end-1}'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206 or not response.headers['Content-Range'].startswith(f'bytes {start}-'):
                    raise RuntimeError('Server must honor exact byte range; refusing full multi-GB download')
                self.cache=response.read();self.cache_start=start
            if len(self.cache)!=fetch_end-start:raise IOError('Incomplete range')
            self.requests+=1;self.transferred+=len(self.cache)
        result=self.cache[start-self.cache_start:end-self.cache_start];self.position=end
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--count-per-domain',type=int,default=4);ap.add_argument('--seed',type=int,default=20261126)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    url='https://zenodo.org/api/records/5706578/files/Val.zip/content'
    remote=RangeFile(url,2425958254);samples=[]
    with zipfile.ZipFile(remote) as archive:
        names=archive.namelist();rng=random.Random(a.seed)
        for domain in ('Rural','Urban'):
            images=sorted(n for n in names if '/'+domain+'/images_png/' in '/'+n and n.endswith('.png'))
            if len(images)<a.count_per_domain:raise ValueError('Unexpected official archive layout')
            chosen=sorted(rng.sample(images,a.count_per_domain))
            for image in chosen:
                label=image.replace('/images_png/','/masks_png/')
                if label not in names:raise ValueError('No validation annotation')
                record=dict(pair_id=domain+'/'+Path(image).stem,domain=domain,split='Val',files={})
                for kind,name in [('image',image),('label',label)]:
                    info=archive.getinfo(name);data=archive.read(name)  # zipfile verifies uncompressed CRC.
                    target=out/domain/kind/Path(name).name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                    record['files'][kind]=dict(path=str(target.relative_to(out)),archive_member=name,
                        bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),zip_CRC32=info.CRC,CRC_verified=True)
                samples.append(record);print('fetched',record['pair_id'],flush=True)
    report=dict(protocol='T05-C-LoveDA-Val-fixed8-v1',official_archive=url,archive_size=remote.size,
        official_dataset_record='https://zenodo.org/records/5706578',dataset_license='CC-BY-4.0',
        seed=a.seed,selection='4 seeded images each Rural/Urban, selected before teacher predictions',
        labels='native PNG IDs 0=ignore/no-data; 1..7=land cover; inspect before evaluation',
        samples=samples,range_requests=remote.requests,downloaded_bytes=remote.transferred,
        whole_archive_checksum_verified=False,member_CRC_verified=True,test_or_train_data_downloaded=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
