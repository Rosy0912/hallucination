"""Attach a preserved AOD resource tree without overwriting packaged code/results."""
import argparse
from pathlib import Path

def link(source,target):
    if not source.exists():raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if target.resolve()!=source.resolve():raise FileExistsError(target)
        return
    target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source.resolve())

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--python-env',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'legacy'
    for name in ['data','models']:link(a.source_root/name,root/name)
    for name in ['llava_separate/pope_popular','llava_hb_imageonly']:
        link(a.source_root/'results'/name/'features.pt',root/'results'/name/'features.pt')
    if a.python_env:link(a.python_env,root/'.venv')
    print('Resources attached; original data/model files unchanged.')
if __name__=='__main__':main()
