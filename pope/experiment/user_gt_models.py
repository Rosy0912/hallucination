"""Controlled GT-readout reproductions of the user's tuning.py and 0.py.

Compare shallow GT MLP / rank-one residual+GRL and capacity-matched deep
GT MLP / dual encoder+GRL+reconstruction. The source cache defines the frozen
backbone and prompt protocol; this is not an exact historical score reproduction.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run import atomic_json,metrics
from aod.core.aod import GradientReversalLayer


def encoder(d,h):return nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,d))
def probe(d,h):return nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,1))


class GTModel(nn.Module):
    def __init__(self,d,variant):
        super().__init__();self.variant=variant
        if variant in ['shallow','projective']:
            self.truth_probe=probe(d,1024)
            if variant=='projective':
                self.v=nn.Parameter(torch.randn(1,d));self.adversary=nn.Sequential(GradientReversalLayer(),probe(d,1024))
        else:
            # Identical initialization for shared truth branch across deep/dual.
            self.enc_t=encoder(d,2048);self.probe_t=probe(d,512)
            if variant=='dual':
                self.enc_h=encoder(d,2048);self.adversary=nn.Sequential(GradientReversalLayer(),probe(d,512))
    def forward(self,x,alpha=1.):
        if self.variant=='shallow':return self.truth_probe(x),None,None,None
        if self.variant=='projective':
            v=nn.functional.normalize(self.v,dim=1);h=(x@v.T)*v;r=x-h
            return self.truth_probe(x-alpha*h),self.adversary(h),r,h
        truth=self.enc_t(x)
        if self.variant=='deep':return self.probe_t(truth),None,None,None
        hallu=self.enc_h(x)
        return self.probe_t(truth+(1-alpha)*hallu),self.adversary(hallu),truth,hallu


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--layers',default='20,24,28');ap.add_argument('--variants',default='shallow,projective,deep,dual')
    ap.add_argument('--alphas',default='0,1,2,3,5',help='Validation strengths for projective inference; training remains alpha=1')
    ap.add_argument('--dual-alphas',default='1',help='Readout of truth+(1-alpha)*hallu; alpha=1 exactly preserves original 0.py')
    ap.add_argument('--tie-policy',choices=['legacy','neutral'],default='legacy')
    ap.add_argument('--seed',type=int,default=42);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    alphas=list(map(float,args.alphas.split(',')));assert alphas and all(np.isfinite(a) for a in alphas)
    dual_alphas=list(map(float,args.dual_alphas.split(',')));assert dual_alphas and all(np.isfinite(a) for a in dual_alphas)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device='cuda' if torch.cuda.is_available() else 'cpu'
    cache=torch.load(args.source/'features.pt',weights_only=False);rows=cache['records'];benchmark=rows[0]['benchmark']
    assert all(r['benchmark']==benchmark for r in rows)
    ids={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','val','test']}
    groups={s:{rows[i]['group'] for i in indices} for s,indices in ids.items()}
    assert not any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
    y=torch.tensor([r['gt'] for r in rows],dtype=torch.float32,device=device)[:,None]
    is_pope=benchmark.startswith('pope_');models=[]
    def predict(scores,indices,threshold):return [dict(rows[i],pred=int(s>=threshold)) for i,s in zip(indices,scores)]
    def rank(cfg):
        m=cfg['validation']
        tie=(-abs(cfg['alpha']-(1. if cfg['variant']=='dual' else 0.)),) if args.tie_policy=='neutral' else ()
        if is_pope:return (m['false_positive_rate']<=8,m['f1'],m['accuracy'],-abs(cfg['threshold']))+tie
        return (m['accuracy'],m['pair_accuracy'],-abs(cfg['threshold']))+tie
    source_sha=hashlib.sha256((args.source/'features.pt').read_bytes()).hexdigest()
    sweep_protocol=dict(source_sha256=source_sha,layers=args.layers,variants=args.variants,seed=args.seed,projective_alphas=alphas,
        dual_alphas=dual_alphas,tie_policy=args.tie_policy,training_alpha=1.,selection='Validation only; fixed test evaluated after selection')
    sweep_path=args.out/'sweep_protocol.json'
    if sweep_path.exists():assert json.loads(sweep_path.read_text())==sweep_protocol
    else:atomic_json(sweep_path,sweep_protocol)
    source_protocol=json.loads((args.source/'extraction_protocol.json').read_text()) if (args.source/'extraction_protocol.json').exists() else {'model':'Qwen/Qwen2.5-VL-7B-Instruct'}
    for layer in map(int,args.layers.split(',')):
        raw=torch.stack(cache['hidden'][layer]).float().to(device)
        mu=raw[ids['train']].mean(0);std=raw[ids['train']].std(0).clamp_min(1e-6)
        for variant in args.variants.split(','):
            tag=f'{benchmark}_L{layer}_{variant}_gt_s{args.seed}';result_path=args.out/(tag+'.validation.json')
            if result_path.exists():models.append(json.loads(result_path.read_text()));continue
            torch.manual_seed(args.seed);net=GTModel(raw.shape[1],variant).to(device)
            standardized=variant in ['deep','dual'];x=(raw-mu)/std if standardized else raw
            epochs=500 if standardized else 100;lr=2e-5 if standardized else 1e-4
            opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4 if standardized else .01)
            history=[];best=None;best_state=None;start=time.time()
            for epoch in range(epochs):
                net.train();opt.zero_grad();out,adv,t,h=net(x[ids['train']])
                main=nn.functional.binary_cross_entropy_with_logits(out,y[ids['train']]);loss=main
                components=dict(main=float(main.detach()))
                if adv is not None:
                    adv_loss=nn.functional.binary_cross_entropy_with_logits(adv,y[ids['train']]);loss=loss+adv_loss
                    components['adversary']=float(adv_loss.detach())
                if variant=='projective':
                    ortho=(t.mul(h).sum(1).square()).mean();loss=loss+.1*ortho;components['orthogonal']=float(ortho.detach())
                if variant=='dual':
                    recon=nn.functional.mse_loss(t+h,x[ids['train']]);loss=loss+recon;components['reconstruction']=float(recon.detach())
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {tag} epoch{epoch}')
                loss.backward()
                if standardized:nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                # Validate on schedule fixed before test, replacing old test-selected epoch.
                if epoch==0 or (epoch+1)%10==0 or epoch+1==epochs:
                    net.eval();local=None
                    with torch.no_grad():
                        for alpha in (alphas if variant=='projective' else dual_alphas if variant=='dual' else [1.]):
                            score=net(x[ids['val']],alpha)[0].flatten().cpu().numpy()
                            for threshold in [-1.,-.5,0.,.5,1.,1.5,2.]:
                                cfg=dict(tag=tag,layer=layer,variant=variant,epoch=epoch+1,alpha=alpha,threshold=threshold,
                                    validation=metrics(predict(score,ids['val'],threshold)))
                                if local is None or rank(cfg)>rank(local):local=cfg
                    history.append(dict(epoch=epoch+1,losses=components,validation=local))
                    if best is None or rank(local)>rank(best):
                        best=local;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                if (epoch+1)%50==0:print(tag,'epoch',epoch+1,'seconds',round(time.time()-start),'best',best['validation']['accuracy'],best['validation']['f1'],flush=True)
            checkpoint=args.out/(tag+'.pt')
            torch.save(dict(state_dict=best_state,variant=variant,layer=layer,mu=mu.cpu() if standardized else None,
                std=std.cpu() if standardized else None,selection=best,source_sha256=source_sha,
                train_ids=[rows[i]['id'] for i in ids['train']]),checkpoint)
            report=best | dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                benchmark=benchmark,history=history,source_sha256=source_sha,
                method='User GT supervision reproduction on frozen prompt features; matching raw MLP ablations; dual is not constrained to be orthogonal',
                feature_protocol=source_protocol,
                training=dict(epochs=epochs,lr=lr,batch_size=len(ids['train']),standardized=standardized,seed=args.seed,
                    float32_matmul_precision='high',parameter_dtype='float32'),
                protocol='Group-disjoint fixed train/val/test; selection by validation only; no test used in training loop')
            atomic_json(result_path,report);models.append(report)
            del net,opt,best_state
            if device=='cuda':torch.cuda.empty_cache()
    # Freeze best layer/epoch/threshold per method before observing test predictions.
    choices={variant:max([m for m in models if m['variant']==variant],key=rank) for variant in args.variants.split(',')}
    selection_path=args.out/'selection.json'
    compact={v:{k:m[k] for k in ['tag','layer','variant','epoch','alpha','threshold','validation','checkpoint','checkpoint_sha256']} for v,m in choices.items()}
    if selection_path.exists():assert json.loads(selection_path.read_text())==compact
    else:atomic_json(selection_path,compact)
    results={};predictions={}
    for variant,cfg in compact.items():
        payload=torch.load(cfg['checkpoint'],weights_only=False)
        assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()==cfg['checkpoint_sha256']
        x=torch.stack([cache['hidden'][cfg['layer']][i] for i in ids['test']]).float().to(device)
        if payload['mu'] is not None:x=(x-payload['mu'].to(device))/payload['std'].to(device)
        net=GTModel(x.shape[1],variant).to(device);net.load_state_dict(payload['state_dict']);net.eval()
        with torch.no_grad():scores=net(x,cfg['alpha'])[0].flatten().cpu().numpy()
        pred=predict(scores,ids['test'],cfg['threshold']);predictions[variant]=pred
        m=metrics(pred);b=metrics([rows[i] for i in ids['test']])
        results[variant]=dict(base=b,selected=m,accuracy_improvement_pp=m['accuracy']-b['accuracy'],f1_improvement_pp=m['f1']-b['f1'],selection=cfg)
        if not is_pope:results[variant]['image_required_subset']=metrics([r for r in pred if r['image']])
        elif all('pope_subset' in r for r in pred):results[variant]['pope_subsets']={s:metrics([r for r in pred if r['pope_subset']==s]) for s in sorted({r['pope_subset'] for r in pred})}
        del net
    atomic_json(args.out/'test_predictions.json',predictions);atomic_json(args.out/'results.json',results)
    print('FROZEN TEST',json.dumps(results),flush=True)


if __name__=='__main__':main()
