"""Extract real-image features, train on disjoint image groups, evaluate Base/CD."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.runtime import ROOT, write_json, sha, binary_metrics, group_split, load_llava, inputs_for, train_direction, provenance
import torch
from aod.data.amber import load_amber_discriminative
from aod.vlm.intervention import resolve_decoder_layers, AODDecodeConfig, aod_next_token_logits
from evaluate import evaluate_records
from aod.vlm.answers import llava_answer_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--images', type=int, default=200, help='0 means all 1004 images')
    p.add_argument('--layer', type=int, default=24)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--extract-only',action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Use a fresh output directory')
    q, ann = a.data_root/'query/query_all.json', a.data_root/'annotations.json'
    records = [dict(id=r.id,image=r.image,question=r.query,gt=int(r.answer=='yes'),typology=r.typology)
               for r in load_amber_discriminative(str(q), str(ann))]
    records = group_split(records, a.images, a.seed)
    for r in records:
        if not (a.image_root/r['image']).is_file():
            raise FileNotFoundError(a.image_root/r['image'])
    a.out.mkdir(parents=True)
    write_json(a.out/'manifest.json',records)
    model, proc = load_llava(a.model)
    yes,no=llava_answer_ids(proc.tokenizer)
    layers, _ = resolve_decoder_layers(model)
    captured = {}
    def hook(_m,_i,out):
        captured['x'] = (out[0] if isinstance(out,tuple) else out)[0,-1].detach().float().cpu()
    handle = layers[a.layer-1].register_forward_hook(hook)
    hidden = []
    start = time.time()
    with torch.inference_mode():
        for i,r in enumerate(records,1):
            x = inputs_for(proc, a.image_root/r['image'], r['question']+'\nAnswer with Yes or No.')
            logits = model(**x,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            r['base_pred'] = int(logits[yes]>logits[no])
            hidden.append(captured['x'])
            if i%100 == 0: print('EXTRACT',i,len(records),round(time.time()-start),flush=True)
    handle.remove()
    hidden = torch.stack(hidden)
    torch.save(dict(records=records,hidden=hidden,layer=a.layer),a.out/'features.pt')
    write_json(a.out/'extraction_protocol.json',dict(answer_ids=[yes,no],scorer='word-start Yes/No; token-fix-v3',images=a.images,layer=a.layer))
    if a.extract_only:return
    tr = [i for i,r in enumerate(records) if r['split']=='train']
    te = [r for r in records if r['split']=='test']
    v, losses = train_direction(hidden[tr], torch.tensor([records[i]['base_pred']==records[i]['gt'] for i in tr]),
                               a.layer,a.out/'aod.pt',a.seed)
    cfg = AODDecodeConfig(layer=a.layer,direction=v,alpha=1.,beta=.5,apc_alpha=.1,mode='cd')
    with torch.inference_mode(), (a.out/'predictions.jsonl').open('w') as f:
        for i,r in enumerate(te,1):
            x = inputs_for(proc,a.image_root/r['image'],r['question']+'\nAnswer with Yes or No.')
            logits,fallback = aod_next_token_logits(model,x,cfg,return_fallback=True)
            scores=logits[0,[yes,no]]
            r['apc_fallback'] = bool(torch.isneginf(scores).all())
            if r['apc_fallback']: scores=fallback[0,[yes,no]]
            r['pred']=int(scores[0]>scores[1])
            f.write(json.dumps(r)+'\n'); f.flush()
            if i%50 == 0: print('EVAL',i,len(te),flush=True)
    result = evaluate_records(te)
    protocol=dict(benchmark='AMBER discriminative',scope='fixed image subset' if a.images else 'all images',
                  seed=a.seed,images=len({r['image'] for r in records}),train_n=len(tr),test_n=len(te),
                  train_images=sorted({r['image'] for r in records if r['split']=='train'}),
                  test_images=sorted({r['image'] for r in te}),layer=a.layer,
                  epochs=5,alpha=1.,beta=.5,apc_alpha=.1,selection='fixed before test; no test tuning',
                  query_sha256=sha(q),annotations_sha256=sha(ann),checkpoint_sha256=sha(a.out/'aod.pt'),
                  training_losses=losses,elapsed_seconds=time.time()-start,**provenance())
    write_json(a.out/'protocol.json',protocol); write_json(a.out/'metrics.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
