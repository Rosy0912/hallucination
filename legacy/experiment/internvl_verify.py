"""Fresh image inference with a frozen, previously validation-selected recipe."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import run as core


def main():
    p=argparse.ArgumentParser()
    p.add_argument('dataset',choices=['hb','pope'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retrain',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    assert not a.out.exists(),'Use a new output directory to guarantee a fresh run'
    source=core.ROOT/'results/internvl3_8b'/a.dataset
    parent=source/'aod_optimized'
    report=json.loads((parent/'results.json').read_text())
    chosen=report['selection'];ckpt=Path(chosen['checkpoint'])
    if ckpt.is_absolute():
        ckpt=core.ROOT/ckpt.relative_to(Path('/home/jmzhang/Workspace/AOD-repro-20260928'))
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==chosen['checkpoint_sha256']
    meta=json.loads(ckpt.with_suffix('.json').read_text())
    rows=json.loads((source/'manifest.json').read_text())
    test=[r for r in rows if r['split']=='test']
    a.out.mkdir(parents=True)
    core.atomic_json(a.out/'frozen_recipe.json',dict(parent=chosen,epoch=meta['selected_epoch'],
        layer=meta['layer'],adv_weight=meta['adv_weight'],seed=42,retrained=a.retrain,
        rule='All settings fixed from prior validation selection; no new parameter search'))
    if a.retrain:
        subprocess.run([sys.executable,'-u',str(core.ROOT/'experiment/train_orthogonal.py'),
            '--source',str(source),'--out',str(a.out/'training'),'--layers',str(meta['layer']),
            '--epochs',str(meta['selected_epoch']),'--adv-weights',str(meta['adv_weight']),
            '--seed','42','--fixed-epoch'],check=True)
        ckpt=next((a.out/'training/checkpoints').glob('*.pt'))
    chosen=dict(chosen,checkpoint=str(ckpt),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())
    selection=a.out/'selection.json';core.atomic_json(selection,chosen)
    args=SimpleNamespace(model=str(core.ROOT/'models/InternVL3-8B-hf'),max_pixels=None,
        out=a.out,layers=[meta['layer']],manifest=source/'manifest.json',checkpoints=str(ckpt),
        alphas=[chosen['alpha']],betas=[chosen['beta']],apc_modes=[chosen['apc_mode']],
        split='test',tag='selected',selection_file=str(selection))
    loaded=core.load_model(args);core.load_model=lambda _:loaded
    core.extract(args,test)
    fresh=torch.load(a.out/'features.pt',weights_only=False)['records']
    original=torch.load(source/'features.pt',weights_only=False)['records']
    expected={r['id']:r['pred'] for r in original if r['split']=='test'}
    assert {r['id']:r['pred'] for r in fresh}==expected,'Fresh Base prediction mismatch'
    core.evaluate(args,fresh)
    key=f"b{chosen['beta']}_{chosen['apc_mode']}"
    path=a.out/f'eval_{ckpt.stem}_test_selected.jsonl'
    predictions=[dict(r,pred=r['predictions'][key]) for r in map(json.loads,path.read_text().splitlines())]
    old_path=parent/f"eval_{Path(report['selection']['checkpoint']).stem}_test_selected.jsonl"
    expected={r['id']:r['predictions'][key] for r in map(json.loads,old_path.read_text().splitlines())}
    assert {r['id']:r['pred'] for r in predictions}==expected,'Fresh AOD prediction mismatch'
    result=dict(base=core.metrics(fresh),selected=core.metrics(predictions),
        complete_test=True,base_predictions_exact=True,selected_predictions_exact=True,
        retrained=a.retrain,selection=chosen,protocol=report['protocol'])
    core.atomic_json(a.out/'metrics.json',result)
    print('FRESH INTERNVL VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__':main()
