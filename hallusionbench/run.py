"""LLaVA HallusionBench: original image-only grouped protocol."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['evaluate','extract','train'])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'legacy';out=a.out.resolve()
    source='results/llava_hb_imageonly'
    commands={
        'evaluate':['experiment/verify_hb_aod_gt.py','--source',source,'--experiment',source+'/aod_gt_balanced_v13',
                    '--model','llava','--out',str(out)],
        'train':['experiment/hb_aod_gt.py','--source',source,'--out',str(out),
                 '--layers','1,4,8,12,16,20,24,28,31,32','--normalization','standard',
                 '--width','64','--balanced','--f1-guard'],
        'extract':['experiment/extract_llava.py','--benchmark','hallusionbench',
                   '--manifest',str(root/'resources/hallusionbench_manifest.json'),
                   '--layers',','.join(map(str,range(1,33))),'--out',str(out)]}
    subprocess.run([sys.executable,'-u',*commands[a.phase]],cwd=root,check=True)
if __name__=='__main__':main()
