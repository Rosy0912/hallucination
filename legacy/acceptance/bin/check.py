"""Read-only acceptance preflight: checkpoints, data splits, images and runtime."""
import collections,hashlib,json,sys,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import torch,transformers
print('ROOT',ROOT);print('PYTHON',sys.executable);print('VERSIONS',platform.python_version(),torch.__version__,transformers.__version__,torch.version.cuda)
for name in ['pope','hb','internvl_pope','llava_hb']:
 config='hb_internvl' if name=='hb' else name
 c=json.loads((ROOT/f'acceptance/configs/{config}.json').read_text())
 checkpoints=c['checkpoint_sha256'] if isinstance(c['checkpoint_sha256'],dict) else {c['checkpoint']:c['checkpoint_sha256']}
 for checkpoint,sha in checkpoints.items():assert hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest()==sha
 cache=torch.load(ROOT/c['source']/'features.pt',weights_only=False);rows=cache['records'];counts=collections.Counter(r['split'] for r in rows)
 assert counts=={'train':c.get('train_n',1800),'val':c.get('val_n',600),'test':c['test_n']}
 g={s:{r['group'] for r in rows if r['split']==s} for s in counts};assert all(not(g[a]&g[b]) for a,b in [('train','val'),('train','test'),('val','test')])
 assert len({r['id'] for r in rows})==len(rows);assert all(r['image'] and (ROOT/'data'/r['image']).is_file() for r in rows)
 print('PASS',name,dict(counts),'all_have_images=True groups_disjoint=True checkpoints_verified='+str(len(checkpoints)))
 del cache
m=ROOT/'models/llava-1.5-7b-hf';v=json.loads((m/'verified_manifest.json').read_text());assert v['revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
print('MODEL',v['repo'],v['revision'])
c=json.loads((ROOT/'acceptance/configs/hb_internvl.json').read_text())
v=json.loads((ROOT/'models/InternVL3-8B-hf/verified_manifest.json').read_text());assert v['revision']==c['revision']
s=json.loads((ROOT/'results/internvl3_8b/hb/aod_optimized/selection.json').read_text())
for key in ['checkpoint_sha256','alpha','beta','apc_mode']:assert c[key]==s[key]
print('MODEL',c['model'],v['revision']);print('PREFLIGHT PASSED')
