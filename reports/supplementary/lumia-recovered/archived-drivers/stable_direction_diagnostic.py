"""Train scalar-probe AOD candidates using training labels; select epoch on validation AUROC only."""
import json,sys,random,torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
R=Path('/home/jmzhang/Workspace/hallucination-20261030');sys.path.insert(0,str(R/'standalone-corrected/amber/src'))
from aod.core.aod import GradientReversalLayer
class StableDirection(nn.Module):
 def __init__(self,x,y):
  super().__init__();self.register_buffer('mean',x.mean(0));self.register_buffer('scale',(x-self.mean).square().mean().sqrt().clamp_min(1e-6));z=(x-self.mean)/self.scale
  self.v=nn.Parameter(F.normalize(z[y>0].mean(0)-z[y==0].mean(0),dim=0));self.probe=nn.Sequential(nn.Linear(1,64),nn.Tanh(),nn.Linear(64,1));self.adv=nn.Sequential(GradientReversalLayer(),nn.Linear(x.shape[1],64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):
  z=(x-self.mean)/self.scale;v=F.normalize(self.v,dim=0);s=z@v[:,None];res=z-s*v[None,:]
  return self.probe(s).flatten(),self.adv(res).flatten()
torch.set_num_threads(4)
for bench in ['amber','chair']:
 root=R/'standalone-corrected'/bench;directory=root/'results/corrected/optimization';out=root/'results/scalar-diagnostic';out.mkdir(exist_ok=False)
 c=torch.load(root/('results/corrected/features/features.pt' if bench=='amber' else 'results/default/features/features.pt'),map_location='cpu',weights_only=False);rows=c['records'];plan=json.loads((directory/'search_plan.json').read_text());trims=set(plan['train_images']) if bench=='amber' else {r['image'] for r in rows if r['split']=='train'};vaims=set(plan['validation_images']);tr=[i for i,r in enumerate(rows) if r['image'] in trims];va=[i for i,r in enumerate(rows) if r['image'] in vaims]
 hidden=c['hidden'].float() if bench=='amber' else torch.stack([x.float() for x in c['hidden'][24]]);x=hidden[tr].cuda();xv=hidden[va].cuda();y=torch.tensor([rows[i].get('base_pred',rows[i].get('pred'))==rows[i]['gt'] for i in tr],device='cuda').float();yv=torch.tensor([rows[i].get('base_pred',rows[i].get('pred'))==rows[i]['gt'] for i in va]).float();weights=torch.where(y>0,.5/y.mean(),.5/(1-y.mean()));torch.manual_seed(42);net=StableDirection(x,y).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);history=[];best=-1
 for epoch in range(100):
  for ids in torch.randperm(len(x),device='cuda').split(256):
   co,ad=net(x[ids]);loss=((F.binary_cross_entropy_with_logits(co,y[ids],reduction='none')+.1*F.binary_cross_entropy_with_logits(ad,y[ids],reduction='none'))*weights[ids]).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
   with torch.no_grad():net.v.copy_(F.normalize(net.v,dim=0))
  if (epoch+1)%5==0:
   with torch.no_grad():co,ad=net(xv);auc=roc_auc_score(yv,co.cpu());adauc=roc_auc_score(yv,ad.cpu())
   history.append(dict(epoch=epoch+1,validation_auroc=auc,residual_auroc=adauc))
   if auc>best:
    best=auc;torch.save(dict(direction=F.normalize(net.v,dim=0).detach().cpu(),state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},epoch=epoch+1,validation_auroc=auc,layer=24,training_ids=[rows[i]['id'] for i in tr],validation_ids=[rows[i]['id'] for i in va],method='AOD scalar consistency probe; centered global-RMS feature scaling; GRL weight0.1; epoch selected by validation AUROC'),out/'direction.pt')
 (out/'training_history.json').write_text(json.dumps(history,indent=2));print(bench,'best_validation_auroc',best,flush=True)
