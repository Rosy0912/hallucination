"""Download the frozen Hugging Face model snapshot and record its identity."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('legacy/models/llava-1.5-7b-hf'))
    a=p.parse_args()
    from huggingface_hub import snapshot_download
    repo='llava-hf/llava-1.5-7b-hf';revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
    snapshot_download(repo_id=repo,revision=revision,local_dir=str(a.output))
    manifest=dict(repo=repo,revision=revision,source='pinned huggingface_hub snapshot_download',
                  files={str(f.relative_to(a.output)):f.stat().st_size for f in a.output.rglob('*')
                         if f.is_file() and '.cache' not in f.parts and f.name!='verified_manifest.json'})
    (a.output/'verified_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
