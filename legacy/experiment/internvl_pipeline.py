"""InternVL3-8B reproduction: immutable image splits, last-prompt features, AOD.

Base and AOD share native HF chat template, BF16 and spelling-variant scoring.
All tuning uses train/validation; the test configuration is locked on disk first.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import run as core
from aod.vlm.intervention import resolve_decoder_layers


def main():
    p=argparse.ArgumentParser()
    p.add_argument('dataset',choices=['pope','hb'])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--stage',choices=['all','extract','aod','validation','finalize'],default='all')
    p.add_argument('--candidate-index',type=int,choices=[0,1])
    a=p.parse_args();torch.set_num_threads(4)
    source=core.ROOT/('results/llava_separate/pope_popular' if a.dataset=='pope' else 'results/llava_hb_imageonly')
    out=core.ROOT/'results/internvl3_8b'/a.dataset
    if a.smoke:out=out/'smoke'
    out.mkdir(parents=True,exist_ok=True)
    old=torch.load(source/'features.pt',weights_only=False)
    rows=[{k:v for k,v in r.items() if k not in ['pred','margin','greedy_token','prompt_style']} for r in old['records']]
    del old
    assert all(r['image'] and (core.ROOT/'data'/r['image']).is_file() for r in rows)
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert all(not groups[x]&groups[y] for x,y in [('train','val'),('train','test'),('val','test')])
    if a.smoke:rows=[next(r for r in rows if r['split']==s) for s in ['train','val']]
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==rows
    else:core.atomic_json(manifest,rows)
    model_path=core.ROOT/'models/InternVL3-8B-hf'
    assert (model_path/'verified_manifest.json').is_file(),'Model hash verification not complete'
    args=SimpleNamespace(model=str(model_path),max_pixels=None,out=out,layers=list(range(1,29)),manifest=manifest)
    protocol=dict(model='OpenGVLab/InternVL3-8B-hf',revision='259a3b64a14623c0ec91a045cb43f7c5af5fa6af',
        dtype='bfloat16',prompt='Official HF chat template; question followed by Answer with Yes or No.',
        token='Last prompt token, decoder block output',layers_one_based=args.layers,
        scorer='max over Yes/yes/space variants vs No/no/space variants; tie Yes',logits_to_keep=1,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_split=str(source),split_counts={s:sum(r['split']==s for r in rows) for s in groups},
        all_samples_have_actual_images=True,torch=torch.__version__,transformers=core.transformers.__version__)
    protocol_path=out/'extraction_protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:core.atomic_json(protocol_path,protocol)
    loaded=core.load_model(args)
    decoder,path=resolve_decoder_layers(loaded.model)
    assert len(decoder)==28
    core.load_model=lambda _:loaded
    print('MODEL READY',path,'layers',len(decoder),flush=True)
    if a.stage in ['all','extract'] or a.smoke:core.extract(args,rows)
    if a.smoke:
        sample=rows[0];inputs=core.inputs_for(loaded,sample)
        direction=torch.ones(3584)
        with torch.inference_mode():
            base=core.forward_next_logits(loaded.model,inputs).float()
            zero=core.forward_next_logits(loaded.model,inputs,layer=20,direction=direction,signed_alpha=0).float()
            steered=core.forward_next_logits(loaded.model,inputs,layer=20,direction=direction,signed_alpha=-1).float()
        assert torch.equal(base,zero) and torch.isfinite(steered).all() and not torch.equal(base,steered)
        core.atomic_json(out/'smoke_passed.json',dict(layer20_alpha0_exact=True,layer20_alpha1_changed=True,
            pixel_values_shape=list(inputs['pixel_values'].shape),input_tokens=inputs['input_ids'].shape[1]))
        print('SMOKE PASSED',flush=True);return
    if a.stage=='extract':return
    cache=torch.load(out/'features.pt',weights_only=False);cached_rows=cache['records'];del cache
    assert {r['id'] for r in rows}=={r['id'] for r in cached_rows}
    dest=out/'aod_optimized';dest.mkdir(exist_ok=True)
    if not (dest/'decoding_candidates.json').exists():
        subprocess.run([sys.executable,'-u',str(core.ROOT/'experiment/train_orthogonal.py'),
            '--source',str(out),'--out',str(dest),'--layers',','.join(map(str,args.layers))],check=True)
    candidates=json.loads((dest/'decoding_candidates.json').read_text())['candidates']
    yes,no=core.get_yes_no_token_ids(loaded.tokenizer)
    audit=[]
    for r in [next(r for r in cached_rows if r['split']==s) for s in ['val','test']]:
        with torch.inference_mode():logits=core.forward_next_logits(loaded.model,core.inputs_for(loaded,r)).float()
        pred=core.score(logits,yes,no);margin=float(logits[0,yes].max()-logits[0,no].max())
        assert pred==r['pred'] and abs(margin-r['margin'])<1e-3
        audit.append(dict(id=r['id'],pred_exact=True,margin_difference=margin-r['margin']))
    core.atomic_json(dest/'base_equivalence.json',audit)
    # Signed alpha pairs repeat within a row. Reuse identical forward outputs,
    # never features/predictions across distinct inputs. This changes no math.
    original_forward=core.forward_next_logits
    memo={'inputs':None,'values':{}}
    def cached_forward(model,inputs,*,layer=None,direction=None,signed_alpha=0.):
        if memo['inputs'] is not inputs:
            memo['inputs']=inputs;memo['values']={}
        k=(layer,float(signed_alpha))
        if k not in memo['values']:
            memo['values'][k]=original_forward(model,inputs,layer=layer,direction=direction,signed_alpha=signed_alpha)
        return memo['values'][k]
    core.forward_next_logits=cached_forward
    args.out=dest;args.alphas=[-1.,-.5,0.,.5,1.];args.betas=[0.,.5,1.,2.];args.apc_modes=['none','fallback','vcd']
    args.split='val';args.tag='';args.selection_file=''
    benchmark=rows[0]['benchmark'];ranking=[]
    if a.stage=='validation':
        assert a.candidate_index is not None,'Validation worker needs an explicit candidate index'
        candidates=[candidates[a.candidate_index]]
    for c in candidates:
        args.checkpoints=c['checkpoint']
        if a.stage!='finalize':core.evaluate(args,[r for r in cached_rows if r['split']=='val'])
        result=json.loads((dest/f"eval_{Path(c['checkpoint']).stem}_val.metrics.json").read_text())
        for k,v in result.items():
            assert v[benchmark+'/val']['n']==sum(r['split']=='val' for r in cached_rows)
            alpha,beta,apc=k.split('_')
            ranking.append(dict(checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],
                alpha=float(alpha[1:]),beta=float(beta[1:]),apc_mode=apc,validation=v[benchmark+'/val']))
    if a.stage=='validation':
        print('VALIDATION CANDIDATE COMPLETE',a.candidate_index,flush=True);return
    def key(c):
        m=c['validation']
        return ((m['false_positive_rate']<=8,m['f1'],m['accuracy'],-abs(c['alpha']),-c['beta']) if a.dataset=='pope'
            else (m['accuracy'],m['pair_accuracy'],-abs(c['alpha']),-c['beta']))
    ranking.sort(key=key,reverse=True);selected=ranking[0];selection=dest/'selection.json'
    if selection.exists():assert json.loads(selection.read_text())==selected
    else:core.atomic_json(selection,selected)
    core.atomic_json(dest/'validation_ranking.json',ranking)
    args.split='test';args.tag='selected';args.selection_file=str(selection);args.checkpoints=selected['checkpoint']
    args.alphas=[selected['alpha']];args.betas=[selected['beta']];args.apc_modes=[selected['apc_mode']]
    test=[r for r in cached_rows if r['split']=='test'];core.evaluate(args,test)
    pred_path=dest/f"eval_{Path(selected['checkpoint']).stem}_test_selected.jsonl"
    pred_key=f"b{selected['beta']}_{selected['apc_mode']}"
    predictions=[dict(r,pred=r['predictions'][pred_key]) for r in map(json.loads,pred_path.read_text().splitlines())]
    assert {r['id'] for r in predictions}=={r['id'] for r in test}
    report=dict(base=core.metrics(test),selected=core.metrics(predictions),selection=selected,protocol=protocol,
        method='Optimized paper-objective orthogonal AOD; last-prompt token; validation-only selection; no GT answer head')
    core.atomic_json(dest/'results.json',report);print('INTERNVL TEST',json.dumps(report),flush=True)

if __name__=='__main__':main()
