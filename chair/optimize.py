"""Select AOD decoding on POPE validation images, then evaluate unseen test images."""
import argparse,json,sys,random,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--cache',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--test-images',type=int,default=50);p.add_argument('--max-new-tokens',type=int,default=128);a=p.parse_args()
sys.path.insert(0,str(a.package));from tools.runtime import *
import torch
from tools.scalar_direction import train_scalar
from aod.core.aod import AODDisentangler,AODCheckpoint,save_checkpoint,load_checkpoint
from aod.vlm.intervention import AODDecodeConfig,greedy_generate_ids
from official_algorithm import CHAIR
from evaluate import score

def train(x,y,epochs,balanced,name):
    torch.manual_seed(42);x=x.cuda().float();y=y.cuda().float().view(-1,1)
    net=AODDisentangler(x.shape[1],512,1.).cuda();net.renormalize_direction_()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=0)
    lossfn=torch.nn.BCEWithLogitsLoss(reduction='none');history=[]
    weights=torch.where(y>0,.5/y.mean(),.5/(1-y.mean())) if balanced else torch.ones_like(y)
    for ep in range(epochs):
        total=0
        for ids in torch.randperm(len(x),device=x.device).split(256):
            z=net(x[ids]);loss=((lossfn(z['pred_consistency'],y[ids])+lossfn(z['pred_residual_adv'],y[ids]))*weights[ids]).mean()
            opt.zero_grad();loss.backward();opt.step();net.renormalize_direction_();total+=loss.item()*len(ids)
        history.append(total/len(x))
    save_checkpoint(str(a.out/(name+'.pt')),AODCheckpoint(24,x.shape[1],512,1.,42),net.cpu().eval())
    return net.v_unit().detach().view(-1),history

def quality(rows,key,e):
    m=score(e,rows,key+'_caption');s=m.pop('sentences');m['mean_tokens']=sum(r[key+'_tokens'] for r in rows)/len(rows)
    m['correct_unique_objects_per_image']=sum(len(set(r['generated_objects'])&set(r['ground_truth'])) for r in s)/len(s)
    m['object_recall']=sum(len(set(r['generated_objects'])&set(r['ground_truth']))/max(1,len(r['ground_truth'])) for r in s)/len(s)
    return m

a.out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
audit=json.loads((a.cache.parent/'extraction_protocol.json').read_text());assert 'token-fix-v3' in audit['scorer'], 'Re-extract corrected consistency labels'
c=torch.load(a.cache,map_location='cpu',weights_only=False);rows=c['records'];tr=[i for i,r in enumerate(rows) if r['split']=='train'];validation=sorted({r['image'] for r in rows if r['split']=='val'});random.Random(20261030).shuffle(validation);validation=validation[:16]
previous=sorted({r['image'] for r in rows if r['split']=='test'});random.Random(42).shuffle(previous);seen=set(previous[:50]);test=sorted(set(previous)-seen)
if not 1<=a.test_images<=len(test):raise ValueError('Choose 1..50 unseen test images')
test=test[:a.test_images]
trainims={rows[i]['image'] for i in tr};assert not trainims&set(validation) and not trainims&set(test) and not set(test)&set(validation)
x=torch.stack([c['hidden'][24][i].float() for i in tr]);y=torch.tensor([rows[i]['pred']==rows[i]['gt'] for i in tr]);vectors={};hist={}
vectors['original5'],hist['original5']=train_direction(x,y,24,a.out/'original5.pt',42,5)
for name,ep,bal in [('normalized5',5,False),('balanced30',30,True)]:vectors[name],hist[name]=train(x,y,ep,bal,name)
va=[i for i,r in enumerate(rows) if r['image'] in set(validation)]
xv=torch.stack([c['hidden'][24][i].float() for i in va]);yv=torch.tensor([rows[i]['pred']==rows[i]['gt'] for i in va])
vectors['scalar'],hist['scalar']=train_scalar(x,y,xv,yv,a.out/'scalar.pt')
del c,x,y,xv,yv
configs=[dict(name='original5',alpha=.5,beta=.5),dict(name='original5',alpha=2.,beta=.5),dict(name='original5',alpha=-1.,beta=.5),dict(name='normalized5',alpha=1.,beta=.5),dict(name='balanced30',alpha=.5,beta=.5),dict(name='balanced30',alpha=1.,beta=.5)]
configs.extend([dict(name='scalar',alpha=alpha,beta=.5) for alpha in [.5,1.,-1.]])
write_json(a.out/'search_plan.json',dict(train_ids=[rows[i]['id'] for i in tr],validation_images=validation,unseen_test_images=test,previous_test_excluded=sorted(seen),configs=configs,training_losses=hist,selection='Both CHAIRs and CHAIRi must improve on validation; correct unique objects >=90% Base and mean tokens >=80% Base; rank CHAIRs then CHAIRi; test scored only after freezing'))
e=CHAIR([int(Path(im).stem.rsplit('_',1)[-1]) for im in validation+test],str(a.annotations));e.get_annotations()
write_json(a.out/'ground_truth.json',{str(k):sorted(v) for k,v in e.imid_to_objects.items()})
model,proc=load_llava(a.model);prompt='Please describe this image in detail.'
def generate(paths,configs,out):
    rows=[]
    with torch.inference_mode(),out.open('w') as f:
        for i,path in enumerate(paths):
            x=inputs_for(proc,a.data_root/path,prompt);r=dict(image=path,image_id=int(Path(path).stem.rsplit('_',1)[-1]))
            for key,cfg in [('base',None)]+[(f'c{j}',c) for j,c in enumerate(configs)]:
                config=None if cfg is None else AODDecodeConfig(layer=24,direction=vectors[cfg['name']],alpha=cfg['alpha'],beta=cfg['beta'],apc_alpha=.1,mode='cd')
                ids=greedy_generate_ids(model,x,cfg=config,max_new_tokens=a.max_new_tokens,eos_token_ids=[proc.tokenizer.eos_token_id]);r[key+'_caption']=proc.tokenizer.decode(ids[0],skip_special_tokens=True).strip();r[key+'_tokens']=ids.shape[1]
            rows.append(r);f.write(json.dumps(r)+'\n');f.flush();print(out.stem,i+1,len(paths),round(time.time()-start),flush=True)
    return rows
vrows=generate(validation,configs,a.out/'validation_predictions.jsonl');base=quality(vrows,'base',e);stats=[dict(config=cfg,metrics=quality(vrows,f'c{i}',e)) for i,cfg in enumerate(configs)];write_json(a.out/'validation_metrics.json',dict(base=base,candidates=stats))
eligible=[s for s in stats if s['metrics']['CHAIRs']<base['CHAIRs'] and s['metrics']['CHAIRi']<base['CHAIRi'] and s['metrics']['correct_unique_objects_per_image']>=.9*base['correct_unique_objects_per_image'] and s['metrics']['mean_tokens']>=.8*base['mean_tokens']]
if not eligible:
    write_json(a.out/'selection.json',dict(selected=None,reason='No validation improvement satisfying quality guards; final test not run'));print('NO VALIDATED IMPROVEMENT',json.dumps(dict(base=base,candidates=stats)),flush=True);sys.exit(0)
selected=min(eligible,key=lambda s:(s['metrics']['CHAIRs'],s['metrics']['CHAIRi']));write_json(a.out/'selection.json',selected);print('SELECTED',json.dumps(selected),flush=True)
trows=generate(test,[selected['config']],a.out/'test_predictions_raw.jsonl')
for r in trows:r['aod_caption']=r.pop('c0_caption');r['aod_tokens']=r.pop('c0_tokens')
(a.out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trows));m={key:quality(trows,key,e) for key in ['base','aod']};write_json(a.out/'metrics.json',m);write_json(a.out/'protocol.json',dict(validation_images=validation,test_images=test,train_n=len(tr),max_new_tokens=a.max_new_tokens,prompt=prompt,selected=selected['config'],elapsed_seconds=time.time()-start,**provenance()));print('FINAL',json.dumps(m),flush=True)
