"""Refit validation-frozen user-AOD heads on train+validation (80% pool).

Layer, epoch count, alpha and threshold remain frozen. Test image groups stay
excluded. This is a separate final-fit protocol, never another test search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from run import atomic_json,metrics
from user_gt_models import GTModel

ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--seed',type=int,default=42)
ap.add_argument('--train-splits',choices=['train,val','train'],default='train,val');args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
choices=json.loads((args.parent/'selection.json').read_text())
cache=torch.load(args.source/'features.pt',weights_only=False);rows=cache['records']
train=[i for i,r in enumerate(rows) if r['split'] in args.train_splits.split(',')];test=[i for i,r in enumerate(rows) if r['split']=='test']
assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in test})
sha=hashlib.sha256((args.source/'features.pt').read_bytes()).hexdigest()
protocol=dict(parent_selection_sha256=hashlib.sha256((args.parent/'selection.json').read_bytes()).hexdigest(),source_sha256=sha,
    train_ids=[rows[i]['id'] for i in train],test_ids=[rows[i]['id'] for i in test],choices=choices,
    rule='Refit from same seed42 initialization on train+val; all hyperparameters and epoch counts fixed by original validation selection; test never used to choose a refit')
if args.seed!=42 or args.train_splits!='train,val':
    protocol.update(seed=args.seed,train_splits=args.train_splits,
        rule='Repeat frozen layer/epoch/alpha/threshold with declared seed and training split; no new validation or test selection')
p=args.out/'protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:atomic_json(p,protocol)
if (args.out/'results.json').exists():raise SystemExit('Completed frozen refit already exists; do not overwrite')
y=torch.tensor([r['gt'] for r in rows],dtype=torch.float32,device='cuda')[:,None]
models={}
for variant,cfg in choices.items():
    cfg=dict(cfg)
    checkpoint_path=Path(cfg['checkpoint'])
    if checkpoint_path.is_absolute():
        cfg['checkpoint']=str(Path(__file__).resolve().parents[1]/checkpoint_path.relative_to(Path('/home/jmzhang/Workspace/AOD-repro-20260928')))
    assert variant in ['shallow','projective','deep','dual']
    assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()==cfg['checkpoint_sha256']
    original=torch.load(cfg['checkpoint'],weights_only=False);assert original['source_sha256']==sha
    path=args.out/f'{variant}.pt'
    if path.exists():models[variant]=path;continue
    torch.manual_seed(args.seed);raw=torch.stack(cache['hidden'][cfg['layer']]).float().cuda()
    standardized=variant in ['deep','dual'];mu=raw[train].mean(0);std=raw[train].std(0).clamp_min(1e-6)
    x=(raw-mu)/std if standardized else raw
    net=GTModel(raw.shape[1],variant).cuda();opt=torch.optim.AdamW(net.parameters(),lr=2e-5 if standardized else 1e-4,weight_decay=1e-4 if standardized else .01)
    for epoch in range(cfg['epoch']):
        net.train();opt.zero_grad();out,adv,t,h=net(x[train]);loss=nn.functional.binary_cross_entropy_with_logits(out,y[train])
        if adv is not None:loss=loss+nn.functional.binary_cross_entropy_with_logits(adv,y[train])
        if variant=='projective':loss=loss+.1*(t.mul(h).sum(1).square()).mean()
        if variant=='dual':loss=loss+nn.functional.mse_loss(t+h,x[train])
        assert torch.isfinite(loss),(variant,epoch)
        loss.backward()
        if standardized:nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mu=mu.cpu() if standardized else None,
        std=std.cpu() if standardized else None,source_sha256=sha,parent_selection=cfg,train_ids=[rows[i]['id'] for i in train]),path)
    models[variant]=path;print('REFIT',variant,cfg['layer'],cfg['epoch'],len(train),flush=True)
    del net,opt,x,raw,original;torch.cuda.empty_cache()
results={};predictions={}
for variant,path in models.items():
    cfg=choices[variant];payload=torch.load(path,weights_only=False);assert payload['source_sha256']==sha and payload['parent_selection']==cfg
    x=torch.stack([cache['hidden'][cfg['layer']][i] for i in test]).float().cuda()
    if payload['mu'] is not None:x=(x-payload['mu'].cuda())/payload['std'].cuda()
    net=GTModel(x.shape[1],variant).cuda();net.load_state_dict(payload['state_dict']);net.eval()
    with torch.no_grad():scores=net(x,cfg['alpha'])[0].flatten().tolist()
    pred=[dict(rows[i],pred=int(s>=cfg['threshold'])) for i,s in zip(test,scores)]
    predictions[variant]=pred;results[variant]=dict(base=metrics([rows[i] for i in test]),selected=metrics(pred),frozen_selection=cfg,
        checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_n=len(train))
    if args.seed==42 and args.train_splits=='train':
        original_predictions=json.loads((args.parent/'test_predictions.json').read_text())[variant]
        assert {r['id']:r['pred'] for r in original_predictions}=={r['id']:r['pred'] for r in pred},'Same-seed repeat changed predictions'
        results[variant]['original_seed_predictions_exact_match']=True
    if rows[0]['benchmark']=='hallusionbench':results[variant]['image_required_subset']=metrics([r for r in pred if r['image']])
atomic_json(args.out/'test_predictions.json',predictions);atomic_json(args.out/'results.json',results)
print('REFIT TEST',json.dumps({k:v['selected'] for k,v in results.items()}),flush=True)
