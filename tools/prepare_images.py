"""Connect user-downloaded COCO/HallusionBench images to the frozen manifests."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--coco-val2014',type=Path,required=True)
    p.add_argument('--hallusionbench-images',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'legacy'
    target=root/'data/images';target.mkdir(parents=True,exist_ok=True)
    count=0
    for name in ['pope','hallusionbench']:
        for r in json.loads((root/'resources'/f'{name}_manifest.json').read_text()):
            if name=='pope':source=a.coco_val2014/Path(r['image']).name
            else:source=a.hallusionbench_images/r['raw']['filename'].removeprefix('./')
            if not source.is_file():raise FileNotFoundError(source)
            dst=root/'data'/r['image']
            if not dst.exists():dst.symlink_to(source.resolve());count+=1
    print('Linked',count,'images')
if __name__=='__main__':main()
