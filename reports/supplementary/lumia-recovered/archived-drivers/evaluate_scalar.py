"""Predeclared scalar-probe AOD validation only. Never evaluates final test data."""
import sys,json,time,os,torch
from pathlib import Path
R=Path('/home/jmzhang/Workspace/hallucination-20261030');bench=sys.argv[1];root=R/'standalone-corrected'/bench;sys.path.insert(0,str(root));from tools.runtime import *
from aod.vlm.intervention import forward_next_logits,AODDecodeConfig,greedy_generate_ids
from aod.vlm.answers import llava_answer_ids
start=time.time();directory=root/'results/corrected/optimization';out=root/'results/scalar-validation';out.mkdir(exist_ok=False);plan=json.loads((directory/'search_plan.json').read_text());ck=torch.load(root/'results/scalar-diagnostic/direction.pt',map_location='cpu',weights_only=False);v=ck['direction'];model,proc=load_llava(R/'legacy/models/llava-1.5-7b-hf')
if bench=='amber':
 from evaluate import evaluate_records
 yes,no=llava_answer_ids(proc.tokenizer);rows=json.loads((directory/'validation_manifest.json').read_text());configs=[dict(name='scalar',alpha=alpha,beta=beta) for alpha in [.25,.5,1.,2.,-.25,-.5,-1.,-2.] for beta in [0.,.5,1.]];preds=[[] for _ in configs]
 write_json(out/'search_plan.json',dict(configs=configs,selection='Combined with v3 validation candidates BEFORE reading any final test scores; same validation accuracy/F1 guard',direction_sha256=sha(root/'results/scalar-diagnostic/direction.pt')))
 with torch.inference_mode():
  for i,r in enumerate(rows):
   x=inputs_for(proc,R/'data/AMBER/image'/r['image'],r['question']+'\nAnswer with Yes or No.');b=forward_next_logits(model,x)[0,[yes,no]].float();r['base_pred']=int(b[0]>b[1]);logits={}
   for alpha in [.25,.5,1.,2.,-.25,-.5,-1.,-2.]:logits[alpha]=forward_next_logits(model,x,layer=24,direction=v,signed_alpha=alpha)[0,[yes,no]].float().cpu()
   for j,cfg in enumerate(configs):
    s=(1+cfg['beta'])*logits[-cfg['alpha']]-cfg['beta']*logits[cfg['alpha']];preds[j].append(int(s[0]>s[1]))
   if (i+1)%50==0:print('VALIDATION',i+1,len(rows),round(time.time()-start),flush=True)
 result=[dict(config=cfg,metrics=evaluate_records([dict(r,pred=pr) for r,pr in zip(rows,preds[j])])) for j,cfg in enumerate(configs)];write_json(out/'validation_results.json',result)
else:
 from official_algorithm import CHAIR
 from evaluate import score
 paths=plan['validation_images'];configs=[dict(name='scalar',alpha=.5,beta=.5),dict(name='scalar',alpha=1.,beta=.5),dict(name='scalar',alpha=-1.,beta=.5)];e=CHAIR([int(Path(x).stem.rsplit('_',1)[-1]) for x in paths],'');gt=json.loads((directory/'ground_truth.json').read_text());e.imid_to_objects={int(k):set(v) for k,v in gt.items()};base_rows=[json.loads(s) for s in (directory/'validation_predictions.jsonl').read_text().splitlines() if s];base_by={r['image']:r for r in base_rows};rows=[]
 write_json(out/'search_plan.json',dict(configs=configs,selection='Combined with v3 validation candidates BEFORE reading final test scores; same CHAIR and content guards',direction_sha256=sha(root/'results/scalar-diagnostic/direction.pt')))
 with torch.inference_mode(),(out/'validation_predictions.jsonl').open('w') as f:
  for i,path in enumerate(paths):
   x=inputs_for(proc,R/'legacy/data'/path,'Please describe this image in detail.');r=dict(image=path,image_id=int(Path(path).stem.rsplit('_',1)[-1]))
   if path in base_by:r.update({k:base_by[path][k] for k in ['base_caption','base_tokens']})
   else:
    ids=greedy_generate_ids(model,x,cfg=None,max_new_tokens=128,eos_token_ids=[proc.tokenizer.eos_token_id]);r.update(base_caption=proc.tokenizer.decode(ids[0],skip_special_tokens=True).strip(),base_tokens=ids.shape[1])
   for j,cfg in enumerate(configs):
    ids=greedy_generate_ids(model,x,cfg=AODDecodeConfig(layer=24,direction=v,alpha=cfg['alpha'],beta=.5,apc_alpha=.1,mode='cd'),max_new_tokens=128,eos_token_ids=[proc.tokenizer.eos_token_id]);r[f'c{j}_caption']=proc.tokenizer.decode(ids[0],skip_special_tokens=True).strip();r[f'c{j}_tokens']=ids.shape[1]
   rows.append(r);f.write(json.dumps(r)+'\n');f.flush();print('VALIDATION',i+1,len(paths),round(time.time()-start),flush=True)
 def quality(key):
  m=score(e,rows,key+'_caption');s=m.pop('sentences');m['mean_tokens']=sum(r[key+'_tokens'] for r in rows)/len(rows);m['correct_unique_objects_per_image']=sum(len(set(r['generated_objects'])&set(r['ground_truth'])) for r in s)/len(s);m['object_recall']=sum(len(set(r['generated_objects'])&set(r['ground_truth']))/max(1,len(r['ground_truth'])) for r in s)/len(s);return m
 write_json(out/'validation_metrics.json',dict(base=quality('base'),candidates=[dict(config=cfg,metrics=quality(f'c{i}')) for i,cfg in enumerate(configs)]))
print('VALIDATION COMPLETE, NO TEST',flush=True)
