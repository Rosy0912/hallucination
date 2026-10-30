"""Split existing all-layer POPE cache without running or changing the backbone."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from run import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    source=args.source/'features.pt'
    digest=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    cache=torch.load(source,weights_only=False)
    parent=json.loads((args.source/'extraction_protocol.json').read_text())
    counts={}
    for subset in ['pope_random','pope_popular','pope_adversarial']:
        ix=[i for i,r in enumerate(cache['records']) if r['pope_subset']==subset]
        rows=[cache['records'][i] for i in ix]
        assert len(rows)==3000 and len({r['id'] for r in rows})==3000
        counts[subset]={s:sum(r['split']==s for r in rows) for s in ['train','val','test']}
        assert counts[subset]==dict(train=1800,val=600,test=600)
        groups={s:{r['group'] for r in rows if r['split']==s} for s in counts[subset]}
        assert not any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
        out=args.out/subset;out.mkdir(parents=True,exist_ok=True)
        protocol=dict(parent,ids=[r['id'] for r in rows],subset=subset,
            parent_features_sha256=digest,selection='Original image-disjoint split retained; models trained separately per subset')
        p=out/'extraction_protocol.json'
        if p.exists():assert json.loads(p.read_text())==protocol
        else:atomic_json(p,protocol)
        if not (out/'features.pt').exists():
            extracted=dict(records=rows,hidden={l:[values[i] for i in ix] for l,values in cache['hidden'].items()})
            temp=out/'features.tmp';torch.save(extracted,temp);temp.replace(out/'features.pt')
        atomic_json(out/'split_verified.json',dict(n=len(rows),counts=counts[subset],groups_disjoint=True,
            parent_features_sha256=digest,features_reused_without_modification=True))
    atomic_json(args.out/'split_summary.json',counts);print(counts,flush=True)


if __name__=='__main__':main()
