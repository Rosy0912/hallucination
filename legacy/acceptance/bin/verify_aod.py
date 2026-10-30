"""Fresh image inference for a frozen AOD configuration with full provenance."""
import argparse,json,hashlib,os,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiment'))
import torch
import run as core
from user_gt_models import GTModel
from aod.core.aod import load_checkpoint
from aod.vlm.intervention import resolve_decoder_layers,layer_steering_hook
from types import SimpleNamespace
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();os.chdir(ROOT)
 cfg=json.loads(a.config.read_text());assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 source=Path(cfg['source']);c=torch.load(source/'features.pt',weights_only=False);rows=c['records'];te=[i for i,r in enumerate(rows) if r['split']=='test'];assert len(te)==cfg['test_n']
 groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','val','test']};assert all(not groups[x]&groups[y] for x,y in [('train','val'),('train','test'),('val','test')]);assert all(r['image'] and (ROOT/'data'/r['image']).is_file() for r in rows)
 ck=Path(cfg['checkpoint']);assert sha(ck)==cfg['checkpoint_sha256'];capture={}
 if cfg['execution'] in ['readout','regularized']:
  payload=torch.load(ck,weights_only=False)
  if cfg['execution']=='regularized':
   from aod_regularized import AOD
   assert cfg['source_sha256']==sha(source/'features.pt');layer=payload['selection']['layer'];net=AOD(c['hidden'][layer][0].numel()).cuda().eval()
  else:
   assert payload['source_sha256']==sha(source/'features.pt');layer=payload['layer'];net=GTModel(c['hidden'][layer][0].numel(),payload['variant']).cuda().eval()
  net.load_state_dict(payload['state_dict'])
 else:
  meta,net=load_checkpoint(str(ck));layer=meta.layer;direction=net.v_unit().view(-1).detach()
 if cfg['model']=='llava':
  from transformers import AutoProcessor,LlavaForConditionalGeneration
  proc=AutoProcessor.from_pretrained(ROOT/'models/llava-1.5-7b-hf',use_fast=False);model=LlavaForConditionalGeneration.from_pretrained(ROOT/'models/llava-1.5-7b-hf',torch_dtype=torch.float16,device_map='cuda:0').eval();proc.patch_size=model.config.vision_config.patch_size;proc.vision_feature_select_strategy=model.config.vision_feature_select_strategy;proc.num_additional_image_tokens=1
  yes=[proc.tokenizer.encode('Yes',add_special_tokens=False)[-1]];no=[proc.tokenizer.encode('No',add_special_tokens=False)[-1]]
  def inputs(r):
   with Image.open(ROOT/'data'/r['image']) as im:image=im.convert('RGB')
   x=proc(text='USER: <image>\n'+r['question']+' ASSISTANT:',images=image,return_tensors='pt');return {k:v.to('cuda:0',dtype=torch.float16 if v.is_floating_point() else v.dtype) for k,v in x.items()}
  def forward(x):return model(**x,use_cache=False).logits[:,-1,:]
  def decision(m):return int(m>0)
 else:
  loaded=core.load_model(SimpleNamespace(model=str(ROOT/'models'/cfg['model_dir']),max_pixels=cfg.get('max_pixels')));model=loaded.model;yes,no=core.get_yes_no_token_ids(loaded.tokenizer)
  inputs=lambda r:core.inputs_for(loaded,r)
  forward=lambda x:core.forward_next_logits(model,x)
  decision=lambda m:int(m>=0)
 handle=None
 if cfg['execution'] in ['readout','regularized']:
  layers,_=resolve_decoder_layers(model)
  def hook(_m,_i,out):capture['x']=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float()
  handle=layers[layer-1].register_forward_hook(hook)
 protocol=dict(method='AOD',configuration=cfg,config_sha256=sha(a.config),source_sha256=sha(source/'features.pt'),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(0),test_ids=[rows[i]['id'] for i in te],code_sha256=sha(__file__))
 core.atomic_json(a.out/'protocol.json',protocol);predictions=[];baseparity=True;featuresparity=True;start=time.time()
 with torch.inference_mode(),(a.out/'predictions.jsonl').open('w') as f:
  for i in te:
   r=rows[i];x=inputs(r);base=forward(x).float();bm=float(base[0,yes].max()-base[0,no].max());bp=decision(bm);baseparity &= bp==r['pred']
   if cfg['execution'] in ['readout','regularized']:
    h=capture['x'];featuresparity &= torch.equal(h.cpu(),c['hidden'][layer][i].float())
    if payload['mu'] is not None:h=(h-payload['mu'].cuda())/payload['std'].cuda()
    if cfg['execution']=='regularized':h=h.clamp(-10,10)
    score=float(net(h[None],cfg['alpha'])[0].item())+cfg.get('base_weight',0)*bm;pred=int(score>=cfg['threshold'])
   else:
    with layer_steering_hook(model,layer,direction,-cfg['alpha']):pos=forward(x).float()
    with layer_steering_hook(model,layer,direction,cfg['alpha']):neg=forward(x).float()
    logits=(1+cfg['beta'])*pos-cfg['beta']*neg
    if cfg['apc_mode']!='none':
     plausible=pos>=pos.max(-1,keepdim=True).values+__import__('math').log(.1);logits=torch.where(plausible,logits,pos if cfg['apc_mode']=='fallback' else float('-inf'))
     if not torch.isfinite(logits[:,yes+no]).any():logits=pos
    score=float(logits[0,yes].max()-logits[0,no].max())+cfg.get('base_weight',0)*bm
    pred=int(score>cfg['threshold']) if cfg.get('comparison')=='>' else int(score>=cfg.get('threshold',0))
   rec=dict(r,base_pred=bp,pred=pred,score=score);predictions.append(rec);f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
   if len(predictions)%25==0:print(cfg['model'],cfg['dataset'],len(predictions),len(te),round(time.time()-start,1),flush=True)
 if handle:handle.remove()
 b=core.metrics([dict(r,pred=r['base_pred']) for r in predictions]);s=core.metrics(predictions);passed=(s['f1']>=90 and s['false_positive_rate']<=10) if cfg['dataset']=='pope' else s['accuracy']-b['accuracy']>=5
 result=dict(method='AOD',base=b,selected=s,accuracy_gain_pp=s['accuracy']-b['accuracy'],f1_gain_pp=s['f1']-b['f1'],criterion_met=passed,complete_test=len(predictions)==cfg['test_n'],base_predictions_exact=baseparity,features_exact=featuresparity,elapsed_seconds=time.time()-start)
 if cfg.get('expected_predictions'):
  old=json.loads(Path(cfg['expected_predictions']).read_text());old=old[cfg['expected_key']] if cfg.get('expected_key') else old;result['selected_predictions_exact']={r['id']:r['pred'] for r in old}=={r['id']:r['pred'] for r in predictions}
 core.atomic_json(a.out/'metrics.json',result);print('RESULT',json.dumps(result),flush=True);assert baseparity,'Base mismatch'
 if 'selected_predictions_exact' in result:assert result['selected_predictions_exact'],'AOD prediction mismatch'
if __name__=='__main__':main()
