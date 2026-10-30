"""AOD with a scalar consistency probe and training-only feature normalization.

The projection remains rank one. A scalar probe avoids LayerNorm erasing its
magnitude; the residual still trains through a gradient reversal adversary.
This is an explicit training variant, not an unchanged paper reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from aod.core.aod import GradientReversalLayer

class StableDirection(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.register_buffer('mean',x.mean(0))
        self.register_buffer('scale',(x-self.mean).square().mean().sqrt().clamp_min(1e-6))
        z=(x-self.mean)/self.scale
        self.v=nn.Parameter(F.normalize(z[y>0].mean(0)-z[y==0].mean(0),dim=0))
        self.probe=nn.Sequential(nn.Linear(1,64),nn.Tanh(),nn.Linear(64,1))
        self.adv=nn.Sequential(GradientReversalLayer(),nn.Linear(x.shape[1],64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,x):
        z=(x-self.mean)/self.scale
        v=F.normalize(self.v,dim=0);s=z@v[:,None];res=z-s*v[None,:]
        return self.probe(s).flatten(),self.adv(res).flatten()

def train_scalar(x,y,xv,yv,output,seed=42):
    x=x.float().cuda();y=y.float().cuda();xv=xv.float().cuda();yv=yv.float().cpu()
    if len(y.unique())!=2 or len(yv.unique())!=2:raise ValueError('Training and validation consistency labels must contain both classes')
    weights=torch.where(y>0,.5/y.mean(),.5/(1-y.mean()))
    torch.manual_seed(seed);net=StableDirection(x,y).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    history=[];best=-1;direction=None
    for epoch in range(100):
        for ids in torch.randperm(len(x),device='cuda').split(256):
            co,ad=net(x[ids])
            loss=((F.binary_cross_entropy_with_logits(co,y[ids],reduction='none')+.1*F.binary_cross_entropy_with_logits(ad,y[ids],reduction='none'))*weights[ids]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite AOD loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():net.v.copy_(F.normalize(net.v,dim=0))
        if (epoch+1)%5==0:
            with torch.no_grad():co,ad=net(xv);auc=roc_auc_score(yv,co.cpu());adauc=roc_auc_score(yv,ad.cpu())
            history.append(dict(epoch=epoch+1,validation_auroc=auc,residual_auroc=adauc))
            if auc>best:
                best=auc;direction=F.normalize(net.v,dim=0).detach().cpu()
                torch.save(dict(direction=direction,state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},epoch=epoch+1,validation_auroc=auc,layer=24,
                    method='AOD scalar probe; training-only centered global RMS; GRL0.1; validation AUROC epoch selection'),output)
    return direction,history
