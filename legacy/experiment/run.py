"""Resumable, group-disjoint AOD reproduction on the frozen Qwen2.5-VL backbone.

Uses official AOD math, hooks and prompts. Caches only prefill final-position
features (the released implementation's readout), not generated answer tokens.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'upstream/src'))
import numpy as np
import torch
import transformers
from PIL import Image
from aod.core.aod import AODDisentangler, AODCheckpoint, save_checkpoint, load_checkpoint
from aod.vlm.loader import load_vlm, build_yes_no_inputs, get_yes_no_token_ids
from aod.vlm.intervention import resolve_decoder_layers, move_tensor_inputs, forward_next_logits


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def metrics(records):
    n = len(records)
    if not n:
        return {'n':0}
    tp = sum(r['pred']==1 and r['gt']==1 for r in records)
    tn = sum(r['pred']==0 and r['gt']==0 for r in records)
    fp = sum(r['pred']==1 and r['gt']==0 for r in records)
    fn = sum(r['pred']!=1 and r['gt']==1 for r in records)
    out = dict(n=n, accuracy=100*(tp+tn)/n, precision=100*tp/max(tp+fp,1),
        recall=100*tp/max(tp+fn,1), f1=200*tp/max(2*tp+fp+fn,1),
        yes_ratio=100*sum(r['pred']==1 for r in records)/n, tp=tp,tn=tn,fp=fp,fn=fn)
    out.update(false_positive_rate=100*fp/max(fp+tn,1),
        false_discovery_rate=100*fp/max(tp+fp,1))
    if all(r['benchmark']=='hallusionbench' for r in records):
        figures, pairs = collections.defaultdict(list), collections.defaultdict(list)
        for r in records:
            raw = r['raw']; correct = r['pred'] == r['gt']
            prefix = tuple(str(raw[k]) for k in ('category','subcategory','set_id'))
            if not(raw['category']=='VS' and str(raw['figure_id'])=='0'):
                figures[prefix+(str(raw['figure_id']),)].append(correct)
            pairs[prefix+(str(raw['question_id']),)].append(correct)
        out.update(figure_accuracy=100*sum(all(v) for v in figures.values())/max(len(figures),1),
            pair_accuracy=100*sum(all(v) for v in pairs.values())/max(len(pairs),1),
            n_figures=len(figures), n_pairs=len(pairs))
    return out


def summary(records):
    groups = collections.defaultdict(list)
    for r in records:
        groups[r['benchmark']+'/'+r['split']].append(r)
        groups[r['benchmark']+'/all'].append(r)
    return {k:metrics(v) for k,v in groups.items()}


def load_model(args):
    model = load_vlm(args.model, device_map='cuda:0', dtype='bf16')
    if args.max_pixels:
        model.processor.image_processor.max_pixels = args.max_pixels
        if hasattr(model.processor.image_processor, 'size'):
            model.processor.image_processor.size['longest_edge'] = args.max_pixels
    return model


def inputs_for(model, row):
    image = Image.open(ROOT/'data'/row['image']).convert('RGB') if row['image'] else None
    question=row['question']
    if row.get('prompt_style')=='literal_visual':
        prefix='Use the actual visible evidence, including unusual or digitally edited details. Do not replace what is shown with what would normally be expected. Inspect the exact objects, colors, counts, spatial relations, and any text relevant to the question.\n' if image is not None else 'Answer the exact question using factual knowledge. Pay attention to qualifiers and negation.\n'
        question=prefix+question
    elif row.get('prompt_style')=='verify_question':
        prefix='Check whether the precise claim asked by the question is true. Use the image as the source of truth rather than assumptions about a typical scene. A surprising detail can still be true if it is visible.\n' if image is not None else 'Check whether the precise claim asked by the question is true. Consider the exact wording rather than a related typical fact.\n'
        question=prefix+question
    return move_tensor_inputs(build_yes_no_inputs(model.processor,model.family,question,image),torch.device('cuda:0'))


def score(logits, yes, no):
    # Same max-over-spelling-variants binary scorer as upstream; not free generation.
    return int((logits[0,yes].max() >= logits[0,no].max()).item())


def centered_forward(model,inputs,layer,direction,signed_alpha,center):
    if center is None:
        return forward_next_logits(model,inputs,layer=layer,direction=direction,signed_alpha=signed_alpha)
    layers,_=resolve_decoder_layers(model)
    def hook(_m,_i,output):
        h=output[0] if isinstance(output,tuple) else output
        z=h[:,-1:,:].float();v=torch.nn.functional.normalize(direction.to(z.device).float(),dim=0)
        shift=((z-center.to(z.device))@v).unsqueeze(-1)*v
        result=h.clone();result[:,-1:,:]=(z+signed_alpha*shift).to(h.dtype)
        return (result,)+output[1:] if isinstance(output,tuple) else result
    handle=layers[layer-1].register_forward_hook(hook)
    try:return forward_next_logits(model,inputs)
    finally:handle.remove()


def extract(args, rows):
    path = args.out/'features.pt'
    cache = torch.load(path,weights_only=False) if path.exists() else {'records':[], 'hidden':{l:[] for l in args.layers}}
    done = {r['id'] for r in cache['records']}
    if all(r['id'] in done for r in rows):
        print('Extraction already complete:',len(rows),flush=True)
        return
    loaded = load_model(args)
    yes, no = get_yes_no_token_ids(loaded.tokenizer)
    layers,_ = resolve_decoder_layers(loaded.model)
    captured = {}
    handles=[]
    for l in args.layers:
        def capture(_m,_i,o, layer=l):
            h=o[0] if isinstance(o,tuple) else o
            captured[layer]=h[0,-1].detach().float().cpu()
        handles.append(layers[l-1].register_forward_hook(capture))
    def save():
        tmp=path.with_suffix('.tmp')
        torch.save(cache,tmp);tmp.replace(path)
        atomic_json(args.out/'base_metrics.json',summary(cache['records']))
    started=time.time()
    try:
        for row in rows:
            if row['id'] in done:continue
            captured.clear()
            with torch.inference_mode():
                logits=forward_next_logits(loaded.model,inputs_for(loaded,row)).float()
            rec=dict(row, pred=score(logits,yes,no),
                margin=float((logits[0,yes].max()-logits[0,no].max()).item()),
                greedy_token=loaded.tokenizer.decode([int(logits.argmax(-1).item())]))
            cache['records'].append(rec)
            for l in args.layers:cache['hidden'][l].append(captured[l])
            if len(cache['records'])%25==0:
                save();print(json.dumps({'event':'extract','n':len(cache['records']),'seconds':round(time.time()-started)}),flush=True)
        save()
    finally:
        for h in handles:h.remove()
    print(json.dumps(summary(cache['records'])),flush=True)


def train(args):
    cache=torch.load(args.out/'features.pt',weights_only=False)
    records=cache['records']
    for benchmark in args.benchmarks.split(','):
        tr=[i for i,r in enumerate(records) if r['benchmark']==benchmark and r['split']=='train']
        va=[i for i,r in enumerate(records) if r['benchmark']==benchmark and r['split']=='val']
        assert tr and va
        assert not({records[i]['group'] for i in tr}&{records[i]['group'] for i in va})
        y=torch.tensor([int(r['pred']==r['gt']) for r in records],dtype=torch.float32).view(-1,1)
        for layer in args.layers:
            x=torch.stack(cache['hidden'][layer]).float()
            for balanced in (False,True):
                torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
                net=AODDisentangler(x.shape[1],512,1.0)
                opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
                weights=None
                if balanced:
                    c=y[tr].sum().item(); assert 0<c<len(tr)
                    weights=torch.where(y[tr]>0, len(tr)/(2*c),len(tr)/(2*(len(tr)-c)))
                logs=[]
                for ep in range(5):
                    net.train()
                    for idx in torch.randperm(len(tr)).split(256):
                        batch=torch.tensor(tr)[idx]
                        output=net(x[batch])
                        losses=[torch.nn.functional.binary_cross_entropy_with_logits(output[k],y[batch],reduction='none')
                                for k in ('pred_consistency','pred_residual_adv')]
                        loss=sum((v*weights[idx]).mean() if balanced else v.mean() for v in losses)
                        opt.zero_grad();loss.backward();opt.step();net.renormalize_direction_()
                    net.eval()
                    with torch.no_grad():
                        out=net(x[va]);pred=(out['pred_consistency']>=0).float();target=y[va]
                        balanced_acc=sum(float((pred[target==c]==c).float().mean()) for c in (0,1))/2
                        logs.append(dict(epoch=ep+1,loss=float(loss),val_probe_accuracy=float((pred==target).float().mean()),
                            val_probe_balanced_accuracy=balanced_acc))
                tag=f'{benchmark}_L{layer}_'+('balanced' if balanced else 'paper')
                ckpt=args.out/'checkpoints'/f'{tag}.pt';ckpt.parent.mkdir(exist_ok=True)
                save_checkpoint(str(ckpt),AODCheckpoint(layer,x.shape[1],512,1.0,args.seed),net)
                atomic_json(ckpt.with_suffix('.json'),dict(tag=tag,train_n=len(tr),val_n=len(va),
                    train_consistency=float(y[tr].mean()),train_ids=[records[i]['id'] for i in tr],
                    validation_ids=[records[i]['id'] for i in va],balanced=balanced,epochs=logs))
                print(tag,logs[-1],flush=True)


def evaluate(args, rows):
    loaded=load_model(args);yes,no=get_yes_no_token_ids(loaded.tokenizer)
    checkpoints=[Path(p) for p in args.checkpoints.split(',')]
    for ckpt in checkpoints:
        meta,net=load_checkpoint(str(ckpt));v=net.v_unit().view(-1).detach()
        center=torch.load(ckpt,weights_only=False).get('center')
        provenance=json.loads(ckpt.with_suffix('.json').read_text())
        train_ids=set(provenance['train_ids'])
        manifest_lookup={r['id']:r['group'] for r in json.loads(args.manifest.read_text())}
        assert train_ids <= manifest_lookup.keys(), 'Cannot verify training groups from this manifest'
        train_groups={manifest_lookup[i] for i in train_ids}
        if args.split in ('val','test'):
            assert not(train_ids & {r['id'] for r in rows})
            assert not(train_groups & {r['group'] for r in rows}), 'Evaluation group overlaps direction training'
        suffix=f'_{args.tag}' if args.tag else ''
        records_path=args.out/f'eval_{ckpt.stem}_{args.split}{suffix}.jsonl'
        if args.split=='test':
            assert len(args.alphas)==len(args.betas)==len(args.apc_modes)==1, 'Test evaluation requires a locked configuration'
            assert args.tag and args.selection_file, 'Test evaluation requires a selection record and unique tag'
            selection=json.loads(Path(args.selection_file).read_text())
            assert selection['checkpoint_sha256']==hashlib.sha256(ckpt.read_bytes()).hexdigest()
            assert selection['alpha']==args.alphas[0] and selection['beta']==args.betas[0] and selection['apc_mode']==args.apc_modes[0]
        signature=dict(checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            model=args.model,max_pixels=args.max_pixels,ids=[r['id'] for r in rows],
            questions_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
            alphas=args.alphas,betas=args.betas,apc_modes=args.apc_modes)
        signature_path=records_path.with_suffix('.config.json')
        if signature_path.exists():
            assert json.loads(signature_path.read_text())==signature, 'Resume configuration mismatch'
        else:atomic_json(signature_path,signature)
        existing=[json.loads(s) for s in records_path.read_text().splitlines()] if records_path.exists() else []
        done={(r['id'],r['alpha']) for r in existing}
        started=time.time()
        with records_path.open('a') as f:
            for row in rows:
                if all((row['id'],a) in done for a in args.alphas):continue
                inputs=inputs_for(loaded,row)
                for alpha in args.alphas:
                    if (row['id'],alpha) in done:continue
                    with torch.inference_mode():
                        pos=centered_forward(loaded.model,inputs,meta.layer,v,-alpha,center).float()
                        neg=centered_forward(loaded.model,inputs,meta.layer,v,alpha,center).float()
                    # One expensive pair of passes supports every beta / APC ablation exactly.
                    pred={}
                    margins={}
                    for beta in args.betas:
                        cd=(1+beta)*pos-beta*neg
                        plausible=pos >= pos.max(-1,keepdim=True).values+np.log(.1)
                        for mode in args.apc_modes:
                            log=cd if mode=='none' else torch.where(plausible,cd,pos if mode=='fallback' else float('-inf'))
                            if not torch.isfinite(log[:,yes+no]).any():log=pos
                            key=f'b{beta}_{mode}'
                            pred[key]=score(log,yes,no)
                            margin=(log[0,yes].max()-log[0,no].max()).item()
                            margins[key]=float(margin) if np.isfinite(margin) else None
                    rec=dict(row,alpha=alpha,predictions=pred,margins=margins,trained_on_sample=row['id'] in train_ids,
                        trained_on_group=row['group'] in train_groups)
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();existing.append(rec)
                if len(existing)%25==0:print(ckpt.stem,len(existing),'records',round(time.time()-started),'seconds',flush=True)
        result={}
        for a in args.alphas:
            for key in existing[0]['predictions']:
                selected=[dict(r,pred=r['predictions'][key]) for r in existing if r['alpha']==a]
                result[f'a{a}_{key}']=summary(selected)
        atomic_json(records_path.with_suffix('.metrics.json'),result)
        print(json.dumps({'checkpoint':ckpt.stem,'configs':len(result),'metrics_file':str(records_path.with_suffix('.metrics.json'))}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['extract','train','eval'])
    ap.add_argument('--model',default=str(ROOT/'models/Qwen2.5-VL-7B-Instruct'))
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/manifest.json')
    ap.add_argument('--out',type=Path,default=ROOT/'results/qwen25vl7b')
    ap.add_argument('--benchmarks',default='hallusionbench,pope_popular')
    ap.add_argument('--layers',type=lambda s:[int(i) for i in s.split(',')],default=[20,24,28])
    ap.add_argument('--alphas',type=lambda s:[float(i) for i in s.split(',')],default=[.5,1.,1.5])
    ap.add_argument('--betas',type=lambda s:[float(i) for i in s.split(',')],default=[0.,.5,1.])
    ap.add_argument('--apc-modes',type=lambda s:s.split(','),default=['none','fallback','vcd'])
    ap.add_argument('--selection-file',default='')
    ap.add_argument('--tag',default='')
    ap.add_argument('--checkpoints',default='')
    ap.add_argument('--split',choices=['all','train','val','test'],default='all')
    ap.add_argument('--max-pixels',type=int,default=1280*28*28)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--prompt-style',choices=['official','literal_visual','verify_question'],default='official')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    rows=json.loads(args.manifest.read_text())
    rows=[r for r in rows if r['benchmark'] in args.benchmarks.split(',') and (args.split=='all' or r['split']==args.split)]
    if args.prompt_style!='official':rows=[dict(r,prompt_style=args.prompt_style) for r in rows]
    if args.limit:rows=rows[:args.limit]
    for r in rows:
        if r['image']:assert (ROOT/'data'/r['image']).is_file(),r['id']
    audit=dict(args=vars(args)|{'out':str(args.out),'manifest':str(args.manifest)},torch=torch.__version__,transformers=transformers.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        n=len(rows),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'upstream/src/aod/core/aod.py',ROOT/'upstream/src/aod/vlm/intervention.py',ROOT/'upstream/src/aod/vlm/loader.py']})
    atomic_json(args.out/f'run_{args.stage}_{int(time.time())}.json',audit)
    if args.stage=='extract':extract(args,rows)
    elif args.stage=='train':train(args)
    else:evaluate(args,rows)


if __name__=='__main__':main()
