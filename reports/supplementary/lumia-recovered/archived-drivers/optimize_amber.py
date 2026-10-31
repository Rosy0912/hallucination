"""Validation-only AOD decoding search, followed by frozen independent evaluation."""
import argparse,json,sys,random,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--image-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.package));from tools.runtime import *
import torch
from aod.core.aod import AODDisentangler,AODCheckpoint,save_checkpoint
from aod.vlm.answers import llava_answer_ids
from aod.vlm.intervention import forward_next_logits
from aod.data.amber import load_amber_discriminative
from evaluate import evaluate_records

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

def metrics(rows,preds):
    return evaluate_records([dict(r,pred=int(v)) for r,v in zip(rows,preds)])

a.out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
c=torch.load(a.cache,map_location='cpu',weights_only=False);rows=c['records'];ims=sorted({r['image'] for r in rows if r['split']=='train'});random.Random(4242).shuffle(ims);val=set(ims[:32]);tr=[i for i,r in enumerate(rows) if r['split']=='train' and r['image'] not in val];va=[i for i,r in enumerate(rows) if r['image'] in val]
vr=[dict(rows[i]) for i in va];write_json(a.out/'validation_manifest.json',vr)
configs=[];vectors={};hist={}
for name,ep,balanced in [('normalized5',5,False),('balanced30',30,True)]:
    vectors[name],hist[name]=train(c['hidden'][tr],torch.tensor([rows[i]['base_pred']==rows[i]['gt'] for i in tr]),ep,balanced,name)
    for alpha in [.25,.5,1.,2.]:
        for sign in [1,-1]:
            for beta in [0.,.5,1.]:configs.append(dict(name=name,alpha=alpha*sign,beta=beta))
write_json(a.out/'search_plan.json',dict(train_images=sorted(set(ims)-val),validation_images=sorted(val),train_n=len(tr),validation_n=len(va),candidates=configs,training_losses=hist,selection='validation accuracy; require no-positive F1 >= Base; ties prefer smaller absolute alpha, beta; test labels never used'))
model,proc=load_llava(a.model);yes,no=llava_answer_ids(proc.tokenizer)
preds={json.dumps(cfg,sort_keys=True):[] for cfg in configs};stats=[]
def margins(x,v,alpha):
    with torch.inference_mode():
        return forward_next_logits(model,x,layer=24,direction=v,signed_alpha=alpha)[0,[yes,no]].float().cpu()
with torch.inference_mode():
    for i,r in enumerate(vr):
        x=inputs_for(proc,a.image_root/r['image'],r['question']+'\nAnswer with Yes or No.')
        base=forward_next_logits(model,x)[0,[yes,no]].float();r['base_pred']=int(base[0]>base[1])
        for name,v in vectors.items():
            for alpha in [.25,.5,1.,2.]:
                neg=margins(x,v,alpha);pos=margins(x,v,-alpha)
                for sign in [1,-1]:
                    po,ne=(pos,neg) if sign==1 else (neg,pos)
                    for beta in [0.,.5,1.]:
                        cfg=dict(name=name,alpha=alpha*sign,beta=beta);scores=(1+beta)*po-beta*ne
                        preds[json.dumps(cfg,sort_keys=True)].append(int(scores[0]>scores[1]))
        if (i+1)%25==0:print('VALIDATION',i+1,len(vr),round(time.time()-start),flush=True)
for cfg in configs:
    key=json.dumps(cfg,sort_keys=True);m=metrics(vr,preds[key]);stats.append(dict(config=cfg,metrics=m))
base=stats[0]['metrics']['base'];eligible=[x for x in stats if x['metrics']['aod']['f1']>=base['f1'] and x['metrics']['aod']['accuracy']>base['accuracy']]
write_json(a.out/'validation_results.json',stats)
if not eligible:
    write_json(a.out/'selection.json',dict(selected=None,base=base,reason='No validation improvement; no final test run'));print('NO VALIDATED IMPROVEMENT',flush=True);sys.exit(0)
chosen=max(eligible,key=lambda x:(x['metrics']['aod']['accuracy'],x['metrics']['aod']['f1'],-abs(x['config']['alpha']),-x['config']['beta']))
write_json(a.out/'selection.json',chosen);print('SELECTED',json.dumps(chosen),flush=True)
allrows=[dict(id=r.id,image=r.image,question=r.query,gt=int(r.answer=='yes'),typology=r.typology) for r in load_amber_discriminative(str(a.data_root/'query/query_all.json'),str(a.data_root/'annotations.json'))]
seen={r['image'] for r in rows};fresh=sorted({r['image'] for r in allrows}-seen);random.Random(20261030).shuffle(fresh);testimages=set(fresh[:200]);test=[r for r in allrows if r['image'] in testimages]
write_json(a.out/'test_manifest.json',test);cfg=chosen['config'];v=vectors[cfg['name']]
with torch.inference_mode(),(a.out/'predictions.jsonl').open('w') as f:
    for i,r in enumerate(test):
        x=inputs_for(proc,a.image_root/r['image'],r['question']+'\nAnswer with Yes or No.')
        b=forward_next_logits(model,x)[0,[yes,no]].float();r['base_pred']=int(b[0]>b[1]);r['base_margin']=float(b[0]-b[1]);pos=margins(x,v,-cfg['alpha']);neg=margins(x,v,cfg['alpha']);s=(1+cfg['beta'])*pos-cfg['beta']*neg;r['pred']=int(s[0]>s[1]);r['aod_margin']=float(s[0]-s[1]);f.write(json.dumps(r)+'\n');f.flush()
        if (i+1)%100==0:print('TEST',i+1,len(test),round(time.time()-start),flush=True)
write_json(a.out/'metrics.json',evaluate_records(test));write_json(a.out/'protocol.json',dict(train_images=len(ims)-len(val),val_images=len(val),test_images=len(testimages),test_overlap=0,selected=cfg,answer_space='conditional Yes/No CD; no full-vocabulary APC mask',elapsed_seconds=time.time()-start,**provenance()))
print('FINAL',json.dumps(evaluate_records(test)),flush=True)
