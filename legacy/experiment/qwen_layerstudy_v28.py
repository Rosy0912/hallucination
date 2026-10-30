"""Last-prompt layer study: raw GT probes, followed by matched AOD probes.

Only development rows enter train stages. The test command loads a previously
frozen selection and does not choose configurations. Layer tables are diagnostic;
the primary layer is selected by grouped development OOF accuracy before test.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from run import ROOT, atomic_json, metrics
from aod.core.aod import AODDisentangler, GradientReversalLayer

LAYERS=list(range(16,25))
SEEDS=[42,43,44]
EPOCHS=[5,10,20,40,80]
ALPHAS=[-1.,-.5,0.,.5,1.]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def head(d,architecture):
    if architecture=='linear':return nn.Linear(d,1)
    return nn.Sequential(nn.Linear(d,64),nn.LayerNorm(64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))


def normalize_fit(x,kind):
    if kind=='standard':return x.mean(0),x.std(0).clamp_min(.01)
    return torch.zeros(x.shape[1],device=x.device),x.square().mean().sqrt().clamp_min(1e-6)


def normalize(x,mean,scale,kind):
    x=(x-mean)/scale
    return x.clamp(-10,10) if kind=='standard' else x


def steer(x,v,alpha):
    if alpha==0:return x
    return x-alpha*(x@v)[:,None]*v


def fit_head(raw,y,valid,cfg,seed,epochs):
    torch.manual_seed(seed)
    mean,scale=normalize_fit(raw,cfg['normalization'])
    x=normalize(raw,mean,scale,cfg['normalization'])
    xv=normalize(valid,mean,scale,cfg['normalization'])
    net=head(x.shape[1],cfg['architecture']).to(x.device)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=.05)
    weights=torch.where(y>0,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
    scores={}
    for epoch in range(1,max(epochs)+1):
        net.train();logits=net(x+.01*torch.randn_like(x)).flatten()
        loss=(nn.functional.binary_cross_entropy_with_logits(logits,.98*y+.01,reduction='none')*weights).mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if epoch in epochs:
            net.eval()
            with torch.inference_mode():scores[epoch]=net(xv).flatten().cpu().numpy()
    return scores,dict(state_dict=cpu_state(net),mean=mean.cpu(),scale=scale.cpu(),seed=seed,
                      architecture=cfg['architecture'],normalization=cfg['normalization'],epoch=max(epochs))


def binary(y,p):
    tp=int(((p==1)&(y==1)).sum());tn=int(((p==0)&(y==0)).sum())
    fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
    return dict(n=len(y),accuracy=100*(tp+tn)/len(y),f1=200*tp/max(1,2*tp+fp+fn),tp=tp,tn=tn,fp=fp,fn=fn)


def select(scores,y,folds,cfgs):
    rankings=[]
    for cfg,s in zip(cfgs,scores):
        for threshold in [-.25,0.,.25]:
            pred=s>=threshold;m=binary(y,pred)
            facc=[float(np.mean(pred[va]==y[va]))*100 for _,va in folds]
            rankings.append(dict(cfg,threshold=threshold,oof=m,fold_accuracy=facc))
    def key(c):
        return (c['oof']['accuracy'],min(c['fold_accuracy']),c['oof']['f1'],
                -abs(c['threshold']),-c.get('epoch',0),-abs(c.get('alpha',0)),c.get('architecture')=='linear')
    return max(rankings,key=key),rankings


def direction(raw,correct,held,hcorrect,layer,train_ids):
    """Paper objectives: consistency on projection; GRL consistency on residual."""
    torch.manual_seed(42)
    scale=raw.square().mean().sqrt().clamp_min(1e-6);x=raw/scale
    net=AODDisentangler(raw.shape[1],512,.1).to(raw.device)
    difference=x[correct==1].mean(0)-x[correct==0].mean(0)
    with torch.no_grad():net.v.copy_(difference[None]);net.renormalize_direction_()
    weights=torch.where(correct>0,len(correct)/(2*correct.sum()),len(correct)/(2*(len(correct)-correct.sum())))
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=0)
    log=[]
    for epoch in range(35):
        net.train();total=0.;grad=0.
        for ids in torch.randperm(len(x),device=x.device).split(256):
            result=net(x[ids]);y=correct[ids,None]
            con=nn.functional.binary_cross_entropy_with_logits(result['pred_consistency'],y,reduction='none')
            adv=nn.functional.binary_cross_entropy_with_logits(result['pred_residual_adv'],y,reduction='none')
            loss=((con+.1*adv).flatten()*weights[ids]).mean()
            opt.zero_grad();loss.backward();grad=float(net.v.grad.norm())
            nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();net.renormalize_direction_();total+=float(loss)*len(ids)
        log.append(dict(epoch=epoch+1,loss=total/len(x),direction_gradient_norm=grad))
    net.eval();v=net.v_unit().detach().flatten()
    with torch.inference_mode():
        r=steer(x,v,1.)
        error=float((r@v).abs().max()/x.norm(dim=-1).max())
        diag=dict(direction_norm=float(v.norm()),relative_orthogonality_error=error,
                  projection_energy_ratio=float((x@v).square().sum()/x.square().sum()))
        assert error<1e-5
        if len(held):
            o=net(held/scale);labels=hcorrect.cpu().numpy()
            for name,k in [('projection','pred_consistency'),('residual','pred_residual_adv')]:
                s=o[k].flatten().cpu().numpy()
                diag[name+'_auroc']=100*roc_auc_score(labels,s)
                diag[name+'_balanced_accuracy']=100*balanced_accuracy_score(labels,s>=0)
    return v,dict(v=v.cpu(),state_dict=cpu_state(net),training_scale=scale.cpu(),layer=layer,
                  train_ids=train_ids,epochs=35,seed=42,adversarial_weight=.1,diagnostics=diag,training_log=log)


def invariants():
    torch.manual_seed(42);x=torch.randn(10,32);v=nn.functional.normalize(torch.randn(32),dim=0)
    assert torch.equal(steer(x,v,0),x)
    assert (steer(x,v,1)@v).abs().max()<1e-5
    z=torch.randn(4,requires_grad=True);(GradientReversalLayer()(z)*torch.tensor([1.,2.,3.,4.])).sum().backward()
    assert torch.equal(z.grad,torch.tensor([-1.,-2.,-3.,-4.]))
    return dict(alpha_zero_identity=True,alpha_one_orthogonal=True,gradient_reversal_sign=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['raw_train','raw_test','orth_train','orth_test'],required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    source=ROOT/'results/qwen_layerstudy_v28/extraction/features.pt';cache=torch.load(source,map_location='cpu',weights_only=False)
    rows=cache['records'];dev=[i for i,r in enumerate(rows) if r['split'] in ['train','val']]
    te=[i for i,r in enumerate(rows) if r['split']=='test'];dr=[rows[i] for i in dev]
    assert len(dev)==755 and len(te)==196 and sorted(cache['hidden'])==LAYERS
    assert not {r['group'] for r in dr}&{rows[i]['group'] for i in te}
    yd=np.array([r['gt'] for r in dr]);bd=np.array([r['pred'] for r in dr]);groups=np.array([r['group'] for r in dr])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=20260928).split(np.zeros(len(dev)),yd,groups))
    for tr,va in folds:assert not set(groups[tr])&set(groups[va])
    ssha=sha(source);device='cuda' if torch.cuda.is_available() else 'cpu'
    orth=a.stage.startswith('orth');out=a.out/('orth' if orth else 'raw')
    if a.stage.endswith('test'):
        cfg=json.loads((out/'selection.json').read_text());assert cfg['source_sha256']==ssha
        assert not (out/'results.json').exists();result=[];allpred={}
        for choice in cfg['layers']:
            layer=choice['layer'];raw=torch.stack([cache['hidden'][layer][i] for i in te]).float()
            if orth:
                ck=torch.load(choice['direction'],map_location='cpu',weights_only=False)
                assert sha(choice['direction'])==choice['direction_sha256']
                assert not set(ck['train_ids'])&{rows[i]['id'] for i in te}
                raw=steer(raw,ck['v'],choice['alpha'])
            logits=[]
            for path in choice['checkpoints']:
                assert sha(path)==choice['checkpoint_sha256'][path]
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert set(ck['train_ids'])=={r['id'] for r in dr}
                h=head(raw.shape[1],ck['architecture']).eval();h.load_state_dict(ck['state_dict'])
                with torch.inference_mode():logits.append(h(normalize(raw,ck['mean'],ck['scale'],ck['normalization'])).flatten().numpy())
            score=np.mean(logits,axis=0);pred=score>=choice['threshold']
            records=[dict(rows[i],base_pred=rows[i]['pred'],pred=int(v),classifier_logit=float(s)) for i,v,s in zip(te,pred,score)]
            b=metrics([rows[i] for i in te]);m=metrics(records)
            result.append(dict(layer=layer,base=b,selected=m,accuracy_gain_pp=m['accuracy']-b['accuracy'],
                               corrected=sum(r['base_pred']!=r['gt'] and r['pred']==r['gt'] for r in records),
                               regressed=sum(r['base_pred']==r['gt'] and r['pred']!=r['gt'] for r in records),configuration=choice))
            allpred[str(layer)]=records
        report=dict(stage=a.stage,primary_layer=cfg['primary_layer'],selection_sha256=sha(out/'selection.json'),
                    source_sha256=ssha,layers=result,primary=next(r for r in result if r['layer']==cfg['primary_layer']))
        atomic_json(out/'test_predictions.json',allpred);atomic_json(out/'results.json',report)
        print('TEST',json.dumps(dict(primary=cfg['primary_layer'],layers=[{k:r[k] for k in ['layer','accuracy_gain_pp','corrected','regressed']} for r in result])),flush=True)
        return
    assert not out.exists();out.mkdir(parents=True)
    protocol=dict(source_sha256=ssha,code_sha256=sha(__file__),development_n=755,test_n=196,layers=LAYERS,
                  seeds=SEEDS,epochs=EPOCHS,alphas=ALPHAS,checks=invariants(),
                  primary_rule='Max OOF accuracy, then worst fold accuracy, F1, smaller threshold; chosen before test',
                  folds=[dict(train_ids=[dr[i]['id'] for i in tr],heldout_ids=[dr[i]['id'] for i in va]) for tr,va in folds],
                  train='755 original train+val questions; grouped 5-fold development CV; 196 test IDs unchanged',
                  probe='GT Yes/No classifier; class-balanced BCE, smoothing .02, feature noise .01, AdamW lr2e-4 wd.05, clip1; three-seed logit mean',
                  direction='35 epochs; projection consistency BCE + .1 residual consistency BCE through GRL; correctness labels; raw-origin projection',
                  comparison='AOD reuses each layer raw-selected architecture/normalization/epoch and retrains head at each alpha; alpha0 OOF scores must match raw',
                  output='Classifier answer accuracy; no Base margin fusion; not a model internal decoding claim')
    atomic_json(out/'protocol.json',protocol);selected=[];oofstore={};rankings=[];diags=[]
    if orth:
        rawcfg=json.loads((a.out/'raw/selection.json').read_text())
        assert rawcfg['source_sha256']==ssha
        parent_oof=np.load(a.out/'raw/oof_scores.npz')
    for layer in LAYERS:
        x=torch.stack([cache['hidden'][layer][i] for i in dev]).float().to(device)
        y=torch.tensor(yd,dtype=torch.float32,device=device);cor=torch.tensor(bd==yd,dtype=torch.float32,device=device)
        candidates={}
        if orth:
            original=next(r for r in rawcfg['layers'] if r['layer']==layer)
            choices=[dict(architecture=original['architecture'],normalization=original['normalization'],alpha=alpha) for alpha in ALPHAS]
            epochs=[original['epoch']]
        else:
            choices=[dict(architecture=h,normalization=n,alpha=0.) for h,n in itertools.product(['linear','mlp64'],['rms','standard'])]
            epochs=EPOCHS
        for fi,(tr,va) in enumerate(folds):
            if orth:
                v,dck=direction(x[tr],cor[tr],x[va],cor[va],layer,[dr[i]['id'] for i in tr])
                path=out/f'direction_L{layer}_f{fi}.pt';torch.save(dck,path)
                diags.append(dict(layer=layer,fold=fi,**dck['diagnostics']))
            else:v=None
            for choice in choices:
                xt=steer(x[tr],v,choice['alpha']);xv=steer(x[va],v,choice['alpha'])
                vals=[]
                for seed in SEEDS:vals.append(fit_head(xt,y[tr],xv,choice,seed,epochs)[0])
                for epoch in epochs:
                    key=f"L{layer}_{choice['architecture']}_{choice['normalization']}_a{choice['alpha']}_e{epoch}"
                    if key not in candidates:candidates[key]=dict(cfg=dict(choice,epoch=epoch,layer=layer),scores=np.zeros(len(dev)))
                    candidates[key]['scores'][va]=np.mean([val[epoch] for val in vals],axis=0)
            print('FOLD',a.stage,layer,fi,flush=True)
        if orth:
            key=f"L{layer}_{original['architecture']}_{original['normalization']}_a0.0_e{original['epoch']}"
            delta=float(np.max(np.abs(candidates[key]['scores']-parent_oof[key])))
            assert delta<1e-5,(key,delta)
            diags.append(dict(layer=layer,alpha_zero_oof_max_difference=delta))
        winner,ranking=select([v['scores'] for v in candidates.values()],yd,folds,[v['cfg'] for v in candidates.values()])
        rankings.extend(ranking);oofstore.update({k:v['scores'] for k,v in candidates.items()})
        if orth:
            v,dck=direction(x,cor,x[:0],cor[:0],layer,[r['id'] for r in dr])
            path=out/f'direction_L{layer}_full.pt';torch.save(dck,path)
            winner.update(direction=str(path),direction_sha256=sha(path))
        trainx=steer(x,v if orth else None,winner['alpha']);paths=[]
        for seed in SEEDS:
            _,ck=fit_head(trainx,y,trainx[:0],winner,seed,[winner['epoch']])
            ck.update(train_ids=[r['id'] for r in dr],source_sha256=ssha,layer=layer,alpha=winner['alpha'])
            path=out/f'answer_L{layer}_s{seed}.pt';torch.save(ck,path);paths.append(str(path))
        winner.update(checkpoints=paths,checkpoint_sha256={path:sha(path) for path in paths})
        selected.append(winner)
        atomic_json(out/'progress.json',selected)
        print('SELECTED LAYER',json.dumps({k:v for k,v in winner.items() if k not in ['checkpoints','checkpoint_sha256']}),flush=True)
    primary=max(selected,key=lambda c:(c['oof']['accuracy'],min(c['fold_accuracy']),c['oof']['f1'],-c['layer']))
    atomic_json(out/'selection.json',dict(source_sha256=ssha,layers=selected,primary_layer=primary['layer'],base_oof=binary(yd,bd)))
    atomic_json(out/'validation_search.json',rankings);atomic_json(out/'direction_diagnostics.json',diags)
    np.savez_compressed(out/'oof_scores.npz',**oofstore)
    print('FROZEN PRIMARY',primary['layer'],primary['oof'],flush=True)


if __name__=='__main__':main()
