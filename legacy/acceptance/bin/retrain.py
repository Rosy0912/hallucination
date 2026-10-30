"""Repeat selected seed42 recipes, using training split only; fresh outputs."""
import argparse,json,sys,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pope','hb']);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-prediction-drift',action='store_true');a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
c=json.loads((ROOT/f'acceptance/configs/{a.dataset}.json').read_text());source=ROOT/c['source']
def run(*args):subprocess.run([sys.executable,'-u',*map(str,args)],check=True)
if a.dataset=='pope':
 parent=a.out/'parent';parent.mkdir();selection=json.loads((source/'gt_sweep/selection.json').read_text());(parent/'selection.json').write_text(json.dumps({'projective':selection['projective']},indent=2));(parent/'test_predictions.json').symlink_to(source/'gt_sweep/test_predictions.json')
 extra=['--allow-prediction-drift'] if a.allow_prediction_drift else []
 run('experiment/refit_user_gt.py','--source',source,'--parent',parent,'--out',a.out/'training','--seed','42','--train-splits','train',*extra)
 checkpoint=a.out/'training/projective.pt'
else:
 run('experiment/train_orthogonal.py','--source',source,'--out',a.out/'training','--layers','16','--epochs','42','--adv-weights','1.0','--fixed-epoch','--seed','42')
 checkpoint=a.out/'training/checkpoints/hallusionbench_L16_orthogonal_scaled_balanced_adv1.0_s42.pt'
run('acceptance/bin/evaluate.py',a.dataset,'--mode','full','--out',a.out/'evaluation','--checkpoint',checkpoint.resolve())
