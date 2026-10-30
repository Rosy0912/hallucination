"""Pool all three MSCOCO POPE subsets while preserving shared-image splits."""
import json
from pathlib import Path
import torch
from run import atomic_json,summary

out=Path('results/aod_pooled_pope');out.mkdir(parents=True,exist_ok=True)
assert not (out/'features.pt').exists(),'Refuse to overwrite experiment'
caches=[torch.load(Path('results')/folder/'features.pt',weights_only=False) for folder in ['pope','pope_random','pope_adversarial']]
layers=sorted(set.intersection(*(set(c['hidden']) for c in caches)))
records=[dict(r,pope_subset=r['benchmark'],benchmark='pope_mscoco') for c in caches for r in c['records']]
assert len(records)==9000 and len({r['id'] for r in records})==9000
assignments={}
for r in records:assert assignments.setdefault(r['group'],r['split'])==r['split']
cache=dict(records=records,hidden={l:[x for c in caches for x in c['hidden'][l]] for l in layers})
torch.save(cache,out/'features.pt')
original=json.loads(Path('data/manifest_official.json').read_text())
manifest=[dict(r,pope_subset=r['benchmark'],benchmark='pope_mscoco') if r['benchmark'].startswith('pope_') else r for r in original]
atomic_json(Path('data/manifest_pooled_pope.json'),manifest)
atomic_json(out/'base_metrics.json',summary(records))
atomic_json(out/'protocol.json',dict(benchmark='POPE MSCOCO pooled random/popular/adversarial',
    train=5400,validation=1800,test=1800,train_images=300,validation_images=100,test_images=100,
    shared_image_groups_preserved=True,metrics='micro F1 and FP/(FP+TN) on pooled test; per-subset secondary',
    source_subsets=['pope_popular','pope_random','pope_adversarial']))
print('Pooled',len(records),'rows;',len(assignments),'images; layers',layers)
