"""Train-normalized consistency head, initialized to the mean-difference sign.
Normalization is only for the classifier; decomposition uses raw-origin vectors.
"""
import torch
from torch import nn
from torch.nn import functional as F
from aod.core.aod import GradientReversalLayer

class StagedDirection(nn.Module):
    def __init__(self,train,correct):
        super().__init__();d=train.shape[1];v=F.normalize((train[correct==1].mean(0)-train[correct==0].mean(0))[None],dim=1)
        self.v=nn.Parameter(v);self.register_buffer('center',train.mean(0));self.register_buffer('scale',((train-self.center)@v.T).std().clamp_min(.01))
        self.consistency=nn.Linear(1,1);nn.init.ones_(self.consistency.weight);nn.init.zeros_(self.consistency.bias)
        self.adversary=nn.Sequential(GradientReversalLayer(),nn.LayerNorm(d),nn.Linear(d,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):
        v=F.normalize(self.v,dim=1);h=(x@v.T)*v;res=x-h
        score=self.consistency(((x-self.center)@v.T)/self.scale).flatten()
        return score,self.adversary(res).flatten()
