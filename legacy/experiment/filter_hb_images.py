"""User scope: train, validate and test HallusionBench with actual images only."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from run import ROOT,atomic_json,summary

ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
cache=torch.load(args.source/'features.pt',weights_only=False)
assert all(r['benchmark']=='hallusionbench' for r in cache['records'])
ix=[i for i,r in enumerate(cache['records']) if r['image']]
rows=[cache['records'][i] for i in ix]
assert len(rows)==951
assert all(str(r['raw']['visual_input']) in ['1','2'] and (ROOT/'data'/r['image']).is_file() for r in rows)
counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']}
assert counts==dict(train=582,val=173,test=196)
groups={s:{r['group'] for r in rows if r['split']==s} for s in counts}
assert not any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
protocol=json.loads((args.source/'extraction_protocol.json').read_text())
with (args.source/'features.pt').open('rb') as stream:parent_hash=hashlib.file_digest(stream,'sha256').hexdigest()
protocol.update(ids=[r['id'] for r in rows],scope='Image-required HallusionBench only, visual_input 1 and 2 in every split',
    parent_features_sha256=parent_hash,split_counts=counts)
p=args.out/'extraction_protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:atomic_json(p,protocol)
if not (args.out/'features.pt').exists():
    filtered=dict(records=rows,hidden={l:[x[i] for i in ix] for l,x in cache['hidden'].items()})
    tmp=args.out/'features.tmp';torch.save(filtered,tmp);tmp.replace(args.out/'features.pt')
atomic_json(args.out/'base_metrics.json',summary(rows))
atomic_json(args.out/'scope_verified.json',dict(n=951,counts=counts,all_images_exist=True,
    visual_input_values=[1,2],split_groups_disjoint=True,features_reused_without_modification=True,parent_features_sha256=parent_hash))
print('IMAGE ONLY',counts,flush=True)
