"""Download public inputs. No private server files or precomputed features required."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
BENCH=json.loads((ROOT/'benchmark.json').read_text())['benchmark']
MODEL='llava-hf/llava-1.5-7b-hf'
REVISION='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size:return path
    error=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'hallucination-benchmark/1.0'})
            with urllib.request.urlopen(req,timeout=60) as response, path.with_suffix(path.suffix+'.part').open('wb') as f:
                shutil.copyfileobj(response,f)
            path.with_suffix(path.suffix+'.part').replace(path);return path
        except Exception as exc:error=exc;time.sleep(attempt+1)
    raise RuntimeError(f'Download failed: {url}; retry prepare or provide local public data: {error}')

def unzip(path,target):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('Corrupt data archive')
        base=Path(target).resolve()
        for name in z.namelist():
            if not (base/name).resolve().is_relative_to(base):raise ValueError('Invalid archive member')
        z.extractall(base)

def link(source,target):
    source=Path(source).resolve();target=Path(target)
    if not source.exists():raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():return
    target.symlink_to(source,target_is_directory=source.is_dir())

def drive(file_id,path):
    if Path(path).exists():return
    import gdown
    if not gdown.download(id=file_id,output=str(path),quiet=False):
        raise RuntimeError('Google Drive download failed; download on an accessible machine and pass a local data-root option.')

def model_complete(path):
    if not (path/'config.json').is_file() or not (path/'preprocessor_config.json').is_file():return False
    if not any((path/name).is_file() for name in ['tokenizer.model','tokenizer.json']):return False
    index=path/'model.safetensors.index.json'
    if not index.is_file():return False
    try:shards=set(json.loads(index.read_text())['weight_map'].values())
    except (ValueError,KeyError):return False
    return bool(shards) and all((path/name).is_file() and (path/name).stat().st_size>0 for name in shards)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path,help='Reuse an already downloaded LLaVA snapshot')
    p.add_argument('--data-only',action='store_true')
    p.add_argument('--coco-root',type=Path,help='Existing COCO val2014 images, optional')
    p.add_argument('--hb-root',type=Path,help='Unpacked HallusionBench image directory, optional')
    p.add_argument('--amber-root',type=Path,help='Existing official AMBER directory containing data/ and image/')
    p.add_argument('--annotations-root',type=Path,help='COCO captions/instances JSON directory, optional')
    a=p.parse_args(argv)
    model=ROOT/'models/llava-1.5-7b-hf'
    if not a.data_only:
        if a.model_dir:
            if not model_complete(a.model_dir):raise ValueError('Local model override is incomplete; supply the full pinned safetensors snapshot')
            link(a.model_dir,model)
        elif not model_complete(model):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=MODEL,revision=REVISION,local_dir=str(model),
                              allow_patterns=['*.json','*.safetensors','tokenizer.model','*.txt'])
    data=ROOT/'data';data.mkdir(exist_ok=True)
    if BENCH in ['pope','hallusionbench','chair']:
        rows=json.loads((ROOT/'assets/manifest.json').read_text())
        if BENCH=='hallusionbench' and not a.hb_root:
            # Public lmms-eval image mirror; immutable revision and byte-level checksum.
            endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
            archive=download(endpoint+'/datasets/lmms-lab-encoder/HallusionBench/resolve/cd417161857aefb23d878d42cf1bb53aa9dd646f/data/image-00000-of-00001.parquet',data/'hb-images.parquet')
            expected='fd46946fa6b3f15ab8c8e2e0dacf54bc801db7915f17a47cec5ca872f30de2a9'
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('HallusionBench parquet checksum mismatch')
            import pyarrow.parquet as pq
            wanted={r['id']:r for r in rows};seen=set()
            for batch in pq.ParquetFile(archive).iter_batches(batch_size=32):
                for record in batch.to_pylist():
                    rid='hb:'+':'.join(str(record[k]) for k in ['category','subcategory','set_id','figure_id','question_id'])
                    if rid in seen:raise ValueError('Duplicate mirror question')
                    reference=wanted[rid];seen.add(rid)
                    if record['question']!=reference['question'] or int(record['gt_answer'])!=reference['gt']:
                        raise ValueError('Mirror questions/labels differ from frozen manifest')
                    dst=data/reference['image'];dst.parent.mkdir(parents=True,exist_ok=True)
                    payload=record['image']['bytes']
                    if dst.exists():
                        if dst.read_bytes()!=payload:raise ValueError('Conflicting image bytes')
                    else:dst.write_bytes(payload)
            if seen!=set(wanted):raise ValueError('Incomplete HallusionBench mirror')
            a.hb_root=data/'images'
        unique={r['image']:r for r in rows}
        def materialize(item):
            rel,r=item;dst=data/rel
            if dst.is_file():return
            if BENCH=='hallusionbench':
                raw=r['raw']['filename'].removeprefix('./').removeprefix('hallusion_bench/')
                candidates=[a.hb_root/raw,a.hb_root/Path(rel).name,a.hb_root/'hallusion_bench'/raw]
                source=next((x for x in candidates if x.is_file()),None)
                if source is None:raise FileNotFoundError(candidates)
                link(source,dst)
            elif a.coco_root:link(a.coco_root/Path(rel).name,dst)
            else:download('https://s3.amazonaws.com/images.cocodataset.org/val2014/'+Path(rel).name,dst)
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(materialize,unique.items()))
        print('Prepared',len(unique),'images and',len(rows),'fixed questions')
    if BENCH=='amber':
        target=data/'AMBER'
        if a.amber_root:link(a.amber_root,target)
        else:
            (target/'data/query').mkdir(parents=True,exist_ok=True)
            base='https://raw.githubusercontent.com/junyangwang0410/AMBER/534babf6bbfcce2e735c26289dedfb21cef3c939/data/'
            download(base+'annotations.json',target/'data/annotations.json')
            download(base+'query/query_all.json',target/'data/query/query_all.json')
            drive('1MaCHgtupcZUjf007anNl4_MV0o4DjXvl',target/'images.zip')
            needed={r['image'] for r in json.loads((target/'data/query/query_all.json').read_text())}
            if not all((target/'image'/name).is_file() for name in needed):unzip(target/'images.zip',target)
        for name in ['data/annotations.json','data/query/query_all.json','image']:
            if not (target/name).exists():raise FileNotFoundError(target/name)
        for name,expected in [('data/annotations.json','db22a436c4836fbb295e60b4e80d519573077579fb7032f01b193c1bd022b797'),('data/query/query_all.json','127680ae61f7d4a92d2f7f07a8deaec07f0689b194d2c7c0f41ead7bd2c5bcf5')]:
            if hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected:raise ValueError('AMBER annotation snapshot mismatch: '+name)
    if BENCH=='chair':
        target=data/'coco/annotations'
        if a.annotations_root:link(a.annotations_root,target)
        elif not target.exists():
            archive=download('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip',data/'coco/annotations.zip')
            unzip(archive,data/'coco')
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab/english/',paths=[str(data/'nltk_data')])
        except LookupError:
            nltk.download('punkt_tab',download_dir=str(data/'nltk_data'),quiet=True,raise_on_error=True)
    (ROOT/'prepared.json').write_text(json.dumps(dict(benchmark=BENCH,model=MODEL,revision=REVISION,
        local_model_override=str(a.model_dir) if a.model_dir else None),indent=2))
    print('PREPARE COMPLETE')

if __name__=='__main__':main()
