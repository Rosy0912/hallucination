"""LLaVA POPE: frozen-image evaluation, extraction, or frozen-recipe retraining."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['evaluate','extract','train'])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'legacy';out=a.out.resolve()
    commands={
        'evaluate':['acceptance/bin/evaluate.py','pope','--mode','full','--out',str(out)],
        'train':['acceptance/bin/retrain.py','pope','--out',str(out)],
        'extract':['experiment/extract_llava.py','--benchmark','pope_mscoco',
                   '--manifest',str(root/'resources/pope_manifest.json'),
                   '--layers','1-32','--out',str(out)]}
    if a.phase=='extract':commands['extract'][commands['extract'].index('1-32')]=','.join(map(str,range(1,33)))
    subprocess.run([sys.executable,'-u',*commands[a.phase]],cwd=root,check=True)
if __name__=='__main__':main()
