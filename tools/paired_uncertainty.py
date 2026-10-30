"""Paired group bootstrap; preserves within-image/question-group dependence."""
import argparse,json
from pathlib import Path
import numpy as np

def binary(rows,positive,group,draws):
    keys=sorted({str(r[group]) for r in rows});lookup={x:i for i,x in enumerate(keys)}
    values=np.zeros((len(keys),2,4),dtype=np.int64)
    for r in rows:
        g=lookup[str(r[group])];gt=r['gt']==positive
        for mode,key in enumerate(['base_pred','pred']):
            pred=r[key]==positive;j=0 if gt and pred else 1 if not gt and pred else 2 if not gt and not pred else 3
            values[g,mode,j]+=1
    def score(v):
        tp,fp,tn,fn=np.moveaxis(v,-1,0)
        return dict(accuracy=100*(tp+tn)/(tp+fp+tn+fn),f1=200*tp/np.maximum(1,2*tp+fp+fn))
    observed=score(values.sum(0));resampled=score(values[draws(len(keys))].sum(1))
    return len(keys),{k:dict(base=float(v[0]),aod=float(v[1]),delta=float(v[1]-v[0]),ci95=np.percentile(resampled[k][:,1]-resampled[k][:,0],[2.5,97.5]).tolist()) for k,v in observed.items()}

def chair(metrics,draws):
    base=metrics['base']['sentences'];aod=metrics['aod']['sentences'];assert [s['image_id'] for s in base]==[s['image_id'] for s in aod]
    values=np.array([[[int(bool(s['hallucinated_objects'])),len(s['hallucinated_objects']),len(s['generated_objects'])] for s in rows] for rows in [base,aod]]).transpose(1,0,2)
    def score(v,n):return dict(CHAIRs=100*v[...,0]/n,CHAIRi=100*v[...,1]/np.maximum(1,v[...,2]))
    n=len(base);observed=score(values.sum(0),n);resampled=score(values[draws(n)].sum(1),n)
    return n,{k:dict(base=float(v[0]),aod=float(v[1]),delta=float(v[1]-v[0]),ci95=np.percentile(resampled[k][:,1]-resampled[k][:,0],[2.5,97.5]).tolist()) for k,v in observed.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=['pope','hallusionbench','amber','chair'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replicates',type=int,default=2000);a=p.parse_args()
    rng=np.random.default_rng(42)
    def draws(n):return rng.integers(0,n,size=(a.replicates,n))
    if a.benchmark=='chair':n,result=chair(json.loads(a.input.read_text()),draws);group='image'
    else:
        rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()];group='group' if a.benchmark=='hallusionbench' else 'image';n,result=binary(rows,0 if a.benchmark=='amber' else 1,group,draws)
    report=dict(method='paired percentile bootstrap at image/group level',seed=42,replicates=a.replicates,groups=n,group_field=group,delta_definition='AOD minus Base; lower is better for CHAIR',metrics=result)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
