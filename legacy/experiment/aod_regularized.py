"""Regularized AOD training, validation selection, and one frozen evaluation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run import atomic_json,metrics
from aod.core.aod import GradientReversalLayer
class AOD(nn.Module):
 def __init__(self,d):
  super().__init__();self.v=nn.Parameter(torch.randn(1,d));self.truth=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,1));self.adv=nn.Sequential(GradientReversalLayer(),nn.Linear(d,128),nn.GELU(),nn.Linear(128,1))
 def forward(self,x,alpha=1.):
  v=nn.functional.normalize(self.v,dim=1);h=(x@v.T)*v;return self.truth(x-alpha*h).flatten(),self.adv(h).flatten()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--layers',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'results.json').exists()
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');dev='cuda';c=torch.load(a.source/'features.pt',weights_only=False);rows=c['records'];ids={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','val','test']};tr=ids['train'];va=ids['val'];te=ids['test'];y=torch.tensor([r['gt'] for r in rows],dtype=torch.float,device=dev);yv=y[va].cpu().numpy();bm=np.array([rows[i]['margin'] for i in va]);baseval=metrics([rows[i] for i in va]);bestkey=None;best=None;beststate=None;bestnorm=None;history=[]
 for layer in map(int,a.layers.split(',')):
  raw=torch.stack(c['hidden'][layer]).float().to(dev);mu=raw[tr].mean(0);std=raw[tr].std(0).clamp_min(.01);x=((raw-mu)/std).clamp(-10,10)
  torch.manual_seed(42);net=AOD(x.shape[1]).to(dev);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01);pos=len(tr)/(2*y[tr].sum());neg=len(tr)/(2*(len(tr)-y[tr].sum()));weights=torch.where(y[tr]>0,pos,neg)
  for epoch in range(1,201):
   net.train()
   for ix in torch.randperm(len(tr),device=dev).split(128):
    indices=torch.tensor(tr,device=dev)[ix];pred,adv=net(x[indices]);loss=((nn.functional.binary_cross_entropy_with_logits(pred,y[indices],reduction='none')+.1*nn.functional.binary_cross_entropy_with_logits(adv,y[indices],reduction='none'))*weights[ix]).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
   if epoch%10:continue
   net.eval();loc=None
   for alpha in [-1.,-.5,0.,.5,1.]:
    with torch.no_grad():score=net(x[va],alpha)[0].cpu().numpy()
    for weight in [0.,.25,1.]:
     for thresh in [-1.,-.5,0.,.5,1.]:
      pp=score+weight*bm>=thresh;tp=int(((pp==1)&(yv==1)).sum());tn=int(((pp==0)&(yv==0)).sum());fp=int(((pp==1)&(yv==0)).sum());fn=int(((pp==0)&(yv==1)).sum());acc=100*(tp+tn)/len(va);f1=200*tp/max(2*tp+fp+fn,1);key=(f1>=baseval['f1'],acc,f1,-weight,-abs(thresh))
      if bestkey is None or key>bestkey:
       bestkey=key;best=dict(method='AOD',layer=layer,epoch=epoch,seed=42,alpha=alpha,base_weight=weight,threshold=thresh,validation=dict(accuracy=acc,f1=f1,n=len(va)));beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};bestnorm=(mu.cpu(),std.cpu())
   history.append(dict(layer=layer,epoch=epoch,best=best.copy()))
  print('FINISHED',layer,'BEST',best,flush=True)
  del net,opt,raw,x
 ck=a.out/'aod.pt';torch.save(dict(state_dict=beststate,mu=bestnorm[0],std=bestnorm[1],selection=best,train_ids=[rows[i]['id'] for i in tr]),ck);best.update(checkpoint=str(ck),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((a.source/'features.pt').read_bytes()).hexdigest(),rule='Validation only; F1 >= validation Base preferred, then accuracy; original grouped split unchanged')
 atomic_json(a.out/'selection.json',best);atomic_json(a.out/'training_history.json',history)
 raw=torch.stack([c['hidden'][best['layer']][i] for i in te]).float().to(dev);x=((raw-bestnorm[0].to(dev))/bestnorm[1].to(dev)).clamp(-10,10);net=AOD(raw.shape[1]).to(dev);net.load_state_dict(beststate);net.eval()
 with torch.no_grad():score=net(x,best['alpha'])[0].cpu().numpy()
 pp=score+best['base_weight']*np.array([rows[i]['margin'] for i in te])>=best['threshold'];pred=[dict(rows[i],base_pred=rows[i]['pred'],pred=int(v)) for i,v in zip(te,pp)];b=metrics([rows[i] for i in te]);s=metrics(pred);result=dict(method='AOD',base=b,selected=s,selection=best,accuracy_gain_pp=s['accuracy']-b['accuracy'],f1_gain_pp=s['f1']-b['f1'],criterion_met=s['accuracy']-b['accuracy']>=5)
 atomic_json(a.out/'test_predictions.json',pred);atomic_json(a.out/'results.json',result);print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
