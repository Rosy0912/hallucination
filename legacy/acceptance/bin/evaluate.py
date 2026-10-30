"""Acceptance evaluation: fresh model forward, immutable checkpoints and test IDs.

--mode cached is POPE-only and reevaluates its head from frozen features.
--mode full always reloads each real image and runs the frozen LLaVA backbone.
"""
import argparse, hashlib, json, os, sys, time, socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiment'))
import torch
from run import metrics, atomic_json
from user_gt_models import GTModel
from aod.core.aod import load_checkpoint
from aod.vlm.intervention import resolve_decoder_layers,layer_steering_hook

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['pope','hb']);ap.add_argument('--mode',choices=['cached','full'],default='full');ap.add_argument('--out',required=True,type=Path);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--checkpoint',type=Path)
    args=ap.parse_args();os.chdir(ROOT);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    assert not args.out.exists(),'Choose a NEW output directory. Historical evidence is never overwritten.'
    assert args.dataset=='pope' or args.mode=='full'
    cfg=json.loads((ROOT/f'acceptance/configs/{args.dataset}.json').read_text())
    path=args.checkpoint or ROOT/cfg['checkpoint']; cksha=sha(path)
    if not args.checkpoint:assert cksha==cfg['checkpoint_sha256']
    source=ROOT/cfg['source'];cache=torch.load(source/'features.pt',weights_only=False);rows=cache['records']
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert all(not(groups[a]&groups[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    test=[i for i,r in enumerate(rows) if r['split']=='test'];assert len(test)==cfg['test_n']
    assert all(r['image'] and (ROOT/'data'/r['image']).is_file() for r in rows)
    if args.limit:test=test[:args.limit]
    args.out.mkdir(parents=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode=='full':assert device=='cuda','Submit full evaluation through Slurm.'
    if args.dataset=='pope':
        payload=torch.load(path,weights_only=False);assert payload['source_sha256']==sha(source/'features.pt')
        net=GTModel(4096,'projective').to(device).eval();net.load_state_dict(payload['state_dict'])
        original=json.loads((source/'gt_sweep/test_predictions.json').read_text())['projective'];expected={r['id']:r['pred'] for r in original}
    else:
        meta,net=load_checkpoint(str(path));v=net.v_unit().view(-1).detach();assert meta.layer==cfg['layer']
        prov=json.loads(path.with_suffix('.json').read_text()); assert set(prov['train_ids'])=={r['id'] for r in rows if r['split']=='train'}
        original=source/'aod_optimized'/cfg['prediction_file'];expected={r['id']:r['predictions']['b2.0_none'] for r in map(json.loads,original.read_text().splitlines())}
    protocol=dict(dataset=args.dataset,mode=args.mode,configuration=cfg,checkpoint=str(path),checkpoint_sha256=cksha,
        feature_sha256=sha(source/'features.pt'),host=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        test_ids=[rows[i]['id'] for i in test],smoke_test=bool(args.limit),retrained_checkpoint=bool(args.checkpoint),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiment/user_gt_models.py',ROOT/'upstream/src/aod/vlm/intervention.py']})
    atomic_json(args.out/'protocol.json',protocol)
    captured={};handle=None
    if args.mode=='full':
        from transformers import AutoProcessor,LlavaForConditionalGeneration
        from PIL import Image
        processor=AutoProcessor.from_pretrained(ROOT/'models/llava-1.5-7b-hf',use_fast=False)
        model=LlavaForConditionalGeneration.from_pretrained(ROOT/'models/llava-1.5-7b-hf',torch_dtype=torch.float16,device_map='cuda:0').eval()
        processor.patch_size=model.config.vision_config.patch_size;processor.vision_feature_select_strategy=model.config.vision_feature_select_strategy;processor.num_additional_image_tokens=1
        yes=processor.tokenizer.encode('Yes',add_special_tokens=False)[-1];no=processor.tokenizer.encode('No',add_special_tokens=False)[-1];assert (yes,no)==(8241,3782)
        if args.dataset=='pope':
            layers,_=resolve_decoder_layers(model)
            def hook(_m,_i,out):captured['x']=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float()
            handle=layers[cfg['layer']-1].register_forward_hook(hook)
    records=[];start=time.time();base_parity=True;head_parity=True;feature_parity=True
    try:
        with (args.out/'predictions.jsonl').open('w') as f,torch.inference_mode():
            for i in test:
                row=rows[i];cached=row['pred']
                if args.mode=='full':
                    with Image.open(ROOT/'data'/row['image']) as im:image=im.convert('RGB')
                    inputs=processor(text='USER: <image>\n'+row['question']+' ASSISTANT:',images=image,return_tensors='pt')
                    inputs={k:t.to('cuda:0',dtype=torch.float16 if t.is_floating_point() else t.dtype) for k,t in inputs.items()}
                    base=model(**inputs,use_cache=False).logits[0,-1].float();base_pred=int(base[yes]>base[no]);base_parity &= base_pred==cached
                    if args.dataset=='pope':
                        x=captured['x'];feature_parity &= torch.equal(x.cpu(),cache['hidden'][cfg['layer']][i].float())
                    else:
                        with layer_steering_hook(model,cfg['layer'],v,-cfg['alpha']):pos=model(**inputs,use_cache=False).logits[0,-1].float()
                        with layer_steering_hook(model,cfg['layer'],v,cfg['alpha']):neg=model(**inputs,use_cache=False).logits[0,-1].float()
                        logits=(1+cfg['beta'])*pos-cfg['beta']*neg;score=float(logits[yes]-logits[no]);pred=int(score>0)
                else:base_pred=cached;x=cache['hidden'][cfg['layer']][i].float().to(device)
                if args.dataset=='pope':score=float(net(x[None],cfg['alpha'])[0].item());pred=int(score>=cfg['threshold'])
                head_parity &= pred==expected[row['id']]
                rec=dict(row,base_pred=base_pred,pred=pred,score=score);records.append(rec);f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                if len(records)%25==0:print(f'{args.dataset} {len(records)}/{len(test)} elapsed={time.time()-start:.1f}s',flush=True)
    finally:
        if handle:handle.remove()
    b=metrics([dict(r,pred=r['base_pred']) for r in records]);a=metrics(records)
    result=dict(base=b,selected=a,accuracy_gain_pp=a['accuracy']-b['accuracy'],f1_gain_pp=a['f1']-b['f1'],
        base_predictions_exact=base_parity,selected_predictions_exact=head_parity,features_exact=feature_parity,
        elapsed_seconds=time.time()-start,complete_test=not args.limit,checkpoint_sha256=cksha,
        criterion_met=(a['f1']>=90 and a['false_positive_rate']<=10) if args.dataset=='pope' else a['accuracy']-b['accuracy']>=5)
    atomic_json(args.out/'metrics.json',result);print(json.dumps(result,indent=2),flush=True)
    assert base_parity,'Base protocol drift: inspect predictions and environment.'
    if not args.checkpoint:assert head_parity,'Frozen checkpoint predictions changed.'
if __name__=='__main__':main()
