"""LLaVA-1.5-7B last-prompt-token cache with explicit historical protocol.

FP16 matches the user's 1.py model precision (earlier extraction used int8).
Default question-only prompt and canonical Yes/No scoring match the earlier
feature-extraction intent, using current processor image-token expansion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from PIL import Image
from transformers import AutoProcessor,LlavaForConditionalGeneration
from run import ROOT,atomic_json,summary
from aod.vlm.intervention import resolve_decoder_layers


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/manifest_pooled_pope.json')
    ap.add_argument('--layers',default='17,21,25,32');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--save-every',type=int,default=100)
    ap.add_argument('--base-reference',type=Path)
    ap.add_argument('--prompt',choices=['legacy','explicit_yes_no'],default='legacy');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    model_path=ROOT/'models/llava-1.5-7b-hf';verified=json.loads((model_path/'verified_manifest.json').read_text())
    assert verified['revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
    rows=[r for r in json.loads(args.manifest.read_text()) if r['benchmark']==args.benchmark]
    if args.limit:rows=rows[:args.limit]
    layer_ids=list(map(int,args.layers.split(',')))
    assert args.save_every>0 and len(set(layer_ids))==len(layer_ids) and all(1<=l<=32 for l in layer_ids)
    reference=torch.load(args.base_reference/'features.pt',weights_only=False) if args.base_reference else None
    reference_ids={r['id']:i for i,r in enumerate(reference['records'])} if reference else None
    config=dict(model=verified['repo'],revision=verified['revision'],dtype='float16',quantization=None,
        processor_use_fast=False,prompt=args.prompt,scorer='canonical Yes/No final token logits; ties predict No',layers_one_based=layer_ids,
        old_code_layers_zero_based=[l-1 for l in layer_ids],readout='raw decoder block output at final prompt token',
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),ids=[r['id'] for r in rows])
    cfgpath=args.out/'extraction_protocol.json'
    if cfgpath.exists():assert json.loads(cfgpath.read_text())==config
    else:atomic_json(cfgpath,config)
    path=args.out/'features.pt';cache=torch.load(path,weights_only=False) if path.exists() else dict(records=[],hidden={l:[] for l in layer_ids})
    done={r['id'] for r in cache['records']}
    def verify_reference(index):
        if reference is None:return
        row=cache['records'][index];old_index=reference_ids[row['id']];old=reference['records'][old_index]
        assert all(row[k]==old[k] for k in ['id','group','split','image','question','gt','pred','margin']),row['id']
        for layer in set(layer_ids)&set(reference['hidden']):
            assert torch.equal(cache['hidden'][layer][index],reference['hidden'][layer][old_index]),(row['id'],layer)
    for index in range(len(cache['records'])):verify_reference(index)
    if all(r['id'] in done for r in rows):print('CACHE COMPLETE',len(rows));return
    processor=AutoProcessor.from_pretrained(model_path,use_fast=False)
    model=LlavaForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.float16,device_map='cuda:0').eval()
    layers,layer_path=resolve_decoder_layers(model);assert len(layers)==32
    tokenizer=processor.tokenizer;yes=tokenizer.encode('Yes',add_special_tokens=False)[-1];no=tokenizer.encode('No',add_special_tokens=False)[-1]
    processor.patch_size=model.config.vision_config.patch_size
    processor.vision_feature_select_strategy=model.config.vision_feature_select_strategy
    processor.num_additional_image_tokens=1
    captured={};handles=[]
    for layer in layer_ids:
        def hook(_m,_i,out,layer=layer):
            h=out[0] if isinstance(out,tuple) else out;captured[layer]=h[0,-1].detach().float().cpu()
        handles.append(layers[layer-1].register_forward_hook(hook))
    def key(r):return (r['image'],r['question'])
    identical={key(r):i for i,r in enumerate(cache['records'])}
    def save():
        temp=path.with_suffix('.tmp');torch.save(cache,temp);temp.replace(path)
        atomic_json(args.out/'base_metrics.json',summary(cache['records']))
    start=time.time();forwards=0;reused=0;conflicting_annotations=[]
    try:
        for row in rows:
            if row['id'] in done:continue
            if key(row) in identical:
                old=identical[key(row)];previous=cache['records'][old]
                assert previous['group']==row['group'] and previous['split']==row['split']
                if previous['gt']!=row['gt']:
                    conflicting_annotations.append(dict(first_id=previous['id'],id=row['id'],first_gt=previous['gt'],gt=row['gt']))
                rec=dict(row,**{k:previous[k] for k in ['pred','margin','greedy_token']},feature_reused_from=previous['id'])
                cache['records'].append(rec)
                for layer in layer_ids:cache['hidden'][layer].append(cache['hidden'][layer][old])
                reused+=1
            else:
                text=row['question']
                if args.prompt=='explicit_yes_no':text+='\nAnswer with Yes or No.'
                prompt=('USER: <image>\n' if row['image'] else 'USER: ')+text+' ASSISTANT:'
                image=Image.open(ROOT/'data'/row['image']).convert('RGB') if row['image'] else None
                inputs=processor(text=prompt,images=image,return_tensors='pt') if image is not None else processor(text=prompt,return_tensors='pt')
                inputs={k:v.to('cuda:0',dtype=torch.float16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
                captured.clear()
                with torch.inference_mode():outputs=model(**inputs,use_cache=False);logits=outputs.logits[0,-1].float()
                assert all(captured[l].shape==(4096,) and torch.isfinite(captured[l]).all() for l in layer_ids)
                if forwards==0:
                    atomic_json(args.out/'input_audit.json',dict(prompt=prompt,token_count=inputs['input_ids'].shape[1],
                        image_token_count=int((inputs['input_ids']==model.config.image_token_index).sum()),
                        yes_id=yes,no_id=no,decoder_path=layer_path,hidden_shape=list(captured[layer_ids[0]].shape),
                        processor_patch_size=processor.patch_size,processor_additional_image_tokens=processor.num_additional_image_tokens))
                rec=dict(row,pred=int(logits[yes]>logits[no]),margin=float(logits[yes]-logits[no]),greedy_token=tokenizer.decode([int(logits.argmax())]))
                identical[key(row)]=len(cache['records']);cache['records'].append(rec)
                for layer in layer_ids:cache['hidden'][layer].append(captured[layer])
                forwards+=1
            done.add(row['id'])
            verify_reference(len(cache['records'])-1)
            if len(cache['records'])%args.save_every==0:
                save();print('EXTRACT',len(cache['records']),'forward',forwards,'reused',reused,'seconds',round(time.time()-start),flush=True)
        save()
    finally:
        for handle in handles:handle.remove()
    atomic_json(args.out/'extraction_completed.json',dict(n=len(cache['records']),forwards_this_run=forwards,reused_this_run=reused,seconds=time.time()-start,
        reference_cache=str(args.base_reference) if reference else None,reference_predictions_and_shared_layers_exact_match=reference is not None,
        conflicting_annotations_this_run=conflicting_annotations,annotation_policy='Original GT labels retained per row, including contradictory labels for identical inputs; GT is never used for inference'))
    print('COMPLETE',json.dumps(summary(cache['records'])),flush=True)


if __name__=='__main__':main()
