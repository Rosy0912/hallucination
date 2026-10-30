"""Train a POPE consistency direction; generate paired held-out COCO captions."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import write_json, sha, train_direction, load_llava, inputs_for, provenance
import torch
from aod.vlm.intervention import AODDecodeConfig, greedy_generate_ids


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pope-features',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--images',type=int,default=50)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--layer',type=int,default=24)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(4)
    cache=torch.load(a.pope_features,map_location='cpu',weights_only=False)
    rows=cache['records'];tr=[i for i,r in enumerate(rows) if r['split']=='train']
    train_images={r['image'] for r in rows if r['split']=='train'}
    images=sorted({r['image'] for r in rows if r['split']=='test'})
    assert not train_images & set(images)
    random.Random(a.seed).shuffle(images)
    if a.images>len(images):raise ValueError('Requested more images than the held-out inventory')
    images=images[:a.images]
    a.out.mkdir(parents=True)
    x=torch.stack([cache['hidden'][a.layer][i].float() for i in tr])
    y=torch.tensor([rows[i]['pred']==rows[i]['gt'] for i in tr])
    v,losses=train_direction(x,y,a.layer,a.out/'aod.pt',a.seed)
    del cache,x,y
    model,proc=load_llava(a.model)
    cfg=AODDecodeConfig(layer=a.layer,direction=v,alpha=1.,beta=.5,apc_alpha=.1,mode='cd')
    eos=[proc.tokenizer.eos_token_id]
    prompt='Please describe this image in detail.'
    start=time.time()
    with torch.inference_mode(),(a.out/'predictions.jsonl').open('w') as f:
        for i,path in enumerate(images,1):
            inputs=inputs_for(proc,a.data_root/path,prompt)
            r=dict(image_id=int(Path(path).stem.rsplit('_',1)[-1]),image=path,prompt=prompt)
            for mode,config in [('base',None),('aod',cfg)]:
                ids=greedy_generate_ids(model,inputs,cfg=config,max_new_tokens=a.max_new_tokens,eos_token_ids=eos)
                r[mode+'_caption']=proc.tokenizer.decode(ids[0],skip_special_tokens=True).strip()
                r[mode+'_tokens']=int(ids.shape[1])
            f.write(json.dumps(r)+'\n');f.flush()
            print('CAPTION',i,len(images),round(time.time()-start),flush=True)
    write_json(a.out/'protocol.json',dict(benchmark='CHAIR',scope='50-image pipeline validation' if a.images==50 else 'fixed held-out image sample',
        images=images,training_image_overlap=0,train_ids=[rows[i]['id'] for i in tr],
        source_features_sha256=sha(a.pope_features),checkpoint_sha256=sha(a.out/'aod.pt'),
        layer=a.layer,seed=a.seed,epochs=5,training_losses=losses,max_new_tokens=a.max_new_tokens,
        prompt=prompt,alpha=1.,beta=.5,apc_alpha=.1,decode='greedy; no KV cache; last-token hook',
        direction='POPE train-only factual-consistency AOD; differs from legacy POPE GT head',
        elapsed_seconds=time.time()-start,**provenance()))

if __name__=='__main__':main()
