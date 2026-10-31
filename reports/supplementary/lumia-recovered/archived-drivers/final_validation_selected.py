"""Freeze the combined validation selection, then run a single paired final evaluation."""
import sys,json,random,time,shutil,torch
from pathlib import Path
R=Path('/home/jmzhang/Workspace/hallucination-20261030');bench=sys.argv[1];root=R/'standalone-corrected'/bench;sys.path.insert(0,str(root));from tools.runtime import *
from aod.vlm.answers import llava_answer_ids
from aod.vlm.intervention import forward_next_logits,AODDecodeConfig,greedy_generate_ids
from aod.core.aod import load_checkpoint
old=root/'results/corrected/optimization';new=root/'results/scalar-validation';out=root/'results/final';out.mkdir(exist_ok=False);plan=json.loads((old/'search_plan.json').read_text())
if bench=='amber':
 stats=json.loads((old/'validation_results.json').read_text())+json.loads((new/'validation_results.json').read_text());base=stats[0]['metrics']['base'];assert all(s['metrics']['base']==base for s in stats)
 eligible=[s for s in stats if s['metrics']['aod']['f1']>=base['f1'] and s['metrics']['aod']['accuracy']>base['accuracy']]
 selected=max(eligible,key=lambda s:(s['metrics']['aod']['accuracy'],s['metrics']['aod']['f1'],-abs(s['config']['alpha']),-s['config']['beta'])) if eligible else None
else:
 oldstats=json.loads((old/'validation_metrics.json').read_text());newstats=json.loads((new/'validation_metrics.json').read_text());base=oldstats['base'];assert newstats['base']==base,(newstats['base'],base)
 stats=oldstats['candidates']+newstats['candidates'];eligible=[s for s in stats if s['metrics']['CHAIRs']<base['CHAIRs'] and s['metrics']['CHAIRi']<base['CHAIRi'] and s['metrics']['correct_unique_objects_per_image']>=.9*base['correct_unique_objects_per_image'] and s['metrics']['mean_tokens']>=.8*base['mean_tokens']]
 selected=min(eligible,key=lambda s:(s['metrics']['CHAIRs'],s['metrics']['CHAIRi'])) if eligible else None
write_json(out/'all_validation_candidates.json',dict(base=base,candidates=stats))
if selected is None:
 write_json(out/'selection.json',dict(selected=None,reason='No candidate met validation improvement/content guards; final test not read'));print('NO VALIDATED IMPROVEMENT',flush=True);sys.exit(0)
cfg=selected['config'];checkpoint=root/'results/scalar-diagnostic/direction.pt' if cfg['name']=='scalar' else old/(cfg['name']+'.pt');shutil.copy2(checkpoint,out/'selected.pt')
if cfg['name']=='scalar':v=torch.load(checkpoint,map_location='cpu',weights_only=False)['direction']
else:_,net=load_checkpoint(str(checkpoint));v=net.v_unit().detach().flatten()
write_json(out/'selection.json',dict(selected=selected,base_validation=base,checkpoint_sha256=sha(out/'selected.pt'),rule=plan['selection'] if 'selection' in plan else 'Validation accuracy with F1 guard; ties smaller intervention',test_access='This selection is frozen before this process constructs or evaluates final test examples. Earlier v3 auto-test output is unused.'))
print('FROZEN',json.dumps(dict(config=cfg,validation=selected['metrics'])),flush=True)
model,proc=load_llava(R/'legacy/models/llava-1.5-7b-hf');start=time.time()
if bench=='amber':
 from aod.data.amber import load_amber_discriminative
 from evaluate import evaluate_records
 yes,no=llava_answer_ids(proc.tokenizer);allrows=[dict(id=r.id,image=r.image,question=r.query,gt=int(r.answer=='yes'),typology=r.typology) for r in load_amber_discriminative(str(R/'data/AMBER/data/query/query_all.json'),str(R/'data/AMBER/data/annotations.json'))];seen={r['image'] for r in json.loads((root/'results/corrected/features/manifest.json').read_text())};fresh=sorted({r['image'] for r in allrows}-seen);random.Random(20261030).shuffle(fresh);testimages=set(fresh[:200]);rows=[r for r in allrows if r['image'] in testimages];assert not testimages&seen
 write_json(out/'test_manifest.json',rows)
 with torch.inference_mode(),(out/'predictions.jsonl').open('w') as f:
  for i,r in enumerate(rows):
   x=inputs_for(proc,R/'data/AMBER/image'/r['image'],r['question']+'\nAnswer with Yes or No.');b=forward_next_logits(model,x)[0,[yes,no]].float();po=forward_next_logits(model,x,layer=24,direction=v,signed_alpha=-cfg['alpha'])[0,[yes,no]].float();ne=forward_next_logits(model,x,layer=24,direction=v,signed_alpha=cfg['alpha'])[0,[yes,no]].float();s=(1+cfg['beta'])*po-cfg['beta']*ne;r.update(base_pred=int(b[0]>b[1]),pred=int(s[0]>s[1]),base_margin=float(b[0]-b[1]),aod_margin=float(s[0]-s[1]));f.write(json.dumps(r)+'\n');f.flush()
   if (i+1)%100==0:print('TEST',i+1,len(rows),round(time.time()-start),flush=True)
 m=evaluate_records(rows)
else:
 from official_algorithm import CHAIR
 from evaluate import score
 paths=plan['unseen_test_images'];rows=[];gt=json.loads((old/'ground_truth.json').read_text());e=CHAIR([int(Path(p).stem.rsplit('_',1)[-1]) for p in paths],'');e.imid_to_objects={int(k):set(v) for k,v in gt.items()};write_json(out/'ground_truth.json',{str(i):sorted(e.imid_to_objects[i]) for i in e.imid_to_objects if i in {int(Path(p).stem.rsplit('_',1)[-1]) for p in paths}})
 config=AODDecodeConfig(layer=24,direction=v,alpha=cfg['alpha'],beta=cfg['beta'],apc_alpha=.1,mode='cd')
 with torch.inference_mode(),(out/'predictions.jsonl').open('w') as f:
  for i,path in enumerate(paths):
   x=inputs_for(proc,R/'legacy/data'/path,'Please describe this image in detail.');r=dict(image=path,image_id=int(Path(path).stem.rsplit('_',1)[-1]))
   for key,c in [('base',None),('aod',config)]:
    ids=greedy_generate_ids(model,x,cfg=c,max_new_tokens=128,eos_token_ids=[proc.tokenizer.eos_token_id]);r[key+'_caption']=proc.tokenizer.decode(ids[0],skip_special_tokens=True).strip();r[key+'_tokens']=ids.shape[1]
   rows.append(r);f.write(json.dumps(r)+'\n');f.flush();print('TEST',i+1,len(paths),round(time.time()-start),flush=True)
 m={}
 for key in ['base','aod']:
  q=score(e,rows,key+'_caption');sentences=q['sentences'];q['mean_tokens']=sum(r[key+'_tokens'] for r in rows)/len(rows);q['correct_unique_objects_per_image']=sum(len(set(s['generated_objects'])&set(s['ground_truth'])) for s in sentences)/len(rows);q['object_recall']=sum(len(set(s['generated_objects'])&set(s['ground_truth']))/max(1,len(s['ground_truth'])) for s in sentences)/len(rows);m[key]=q
write_json(out/'metrics.json',m);write_json(out/'protocol.json',dict(benchmark=bench,selection=cfg,train_validation_plan=plan,final_n=len(rows),final_images=len({r['image'] for r in rows}),test_selection_seed=20261030 if bench=='amber' else 42,answer_ids=[3869,1939],max_new_tokens=128 if bench=='chair' else 1,elapsed_seconds=time.time()-start,runner_sha256=sha(Path(__file__)),**provenance()))
print('COMPLETE',json.dumps({k:{a:b for a,b in v.items() if a!='sentences'} for k,v in m.items() if k in ['base','aod']}),flush=True)
