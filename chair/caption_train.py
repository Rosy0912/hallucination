"""Experimental in-domain direction: collect only train/validation caption features."""
import argparse,json,random,sys,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--images',type=int,default=60);a=p.parse_args()
a.package=a.package.resolve();sys.path.insert(0,str(a.package));from tools.runtime import *
import torch,nltk
from nltk.tokenize import TreebankWordTokenizer,PunktTokenizer
from aod.vlm.intervention import resolve_decoder_layers,greedy_generate_ids
from tools.scalar_direction import train_scalar
from official_algorithm import CHAIR

def image_id(path):return int(Path(path).stem.rsplit('_',1)[-1])
def word_spans(text):
 result=[]
 for lo,hi in PunktTokenizer('english').span_tokenize(text):result.extend((lo+i,lo+j) for i,j in TreebankWordTokenizer().span_tokenize(text[lo:hi]))
 return result

a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.time()
manifest=json.loads((a.package/'assets/manifest.json').read_text())
train=sorted({x['image'] for x in manifest if x['split']=='train'});random.Random(20261030).shuffle(train);train=train[:a.images]
validation=sorted({x['image'] for x in manifest if x['split']=='val'});random.Random(20261030).shuffle(validation);validation=validation[:16]
assert not set(train)&set(validation)
write_json(a.out/'search_plan.json',dict(train_images=train,validation_images=validation,test_access=False,layer=24,target='object mentions supported by COCO annotations; generated train captions',direction='scalar consistency probe with residual adversary'))
e=CHAIR([image_id(x) for x in train+validation],str(a.annotations));e.get_annotations()
model,proc=load_llava(a.model);layers,_=resolve_decoder_layers(model);features=[];records=[];captions=[];skipped=[];prompt='Please describe this image in detail.'
with torch.inference_mode(),(a.out/'captions.jsonl').open('w') as out:
 for split,paths in [('train',train),('validation',validation)]:
  for i,path in enumerate(paths):
   x=inputs_for(proc,a.data_root/path,prompt)
   ids=greedy_generate_ids(model,x,cfg=None,max_new_tokens=128,eos_token_ids=[proc.tokenizer.eos_token_id])[0].tolist()
   if ids and ids[-1]==proc.tokenizer.eos_token_id:ids=ids[:-1]
   caption=proc.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
   row=dict(image=path,image_id=image_id(path),split=split,base_caption=caption,base_tokens=len(ids));captions.append(row);out.write(json.dumps(row)+'\n');out.flush()
   # Exact generated IDs, with prefix-decoded character offsets. This avoids
   # re-tokenizing stripped captions into different word-start token IDs.
   prefixes=[proc.tokenizer.decode(ids[:j],skip_special_tokens=True,clean_up_tokenization_spaces=False) for j in range(len(ids)+1)]
   if any(not caption.startswith(t) for t in prefixes):
    skipped.append(dict(image=path,reason='non-prefix unicode decoding'));continue
   offsets=[(len(prefixes[j]),len(prefixes[j+1])) for j in range(len(ids))]
   tokens=nltk.word_tokenize(caption.lower());spans=word_spans(caption.lower())
   if len(tokens)!=len(spans) or ('toilet' in tokens and 'seat' in tokens):
    skipped.append(dict(image=path,reason='CHAIR word-span alignment edge case'));continue
   words,nodes,indices,_=e.caption_to_words(caption);captured={}
   def hook(_m,_i,o):captured['hidden']=(o[0] if isinstance(o,tuple) else o)[0].detach().float().cpu()
   handle=layers[23].register_forward_hook(hook)
   prompt_len=x['input_ids'].shape[1];x['input_ids']=torch.cat([x['input_ids'],torch.tensor([ids],device=x['input_ids'].device)],dim=1);x['attention_mask']=torch.ones_like(x['input_ids'])
   try:model(**x,use_cache=False,logits_to_keep=1)
   finally:handle.remove()
   hidden=captured['hidden'];assert hidden.shape[0]==x['input_ids'].shape[1]
   for word,node,index in zip(words,nodes,indices):
    lo,hi=spans[index];matching=[j for j,(l,h) in enumerate(offsets) if h>lo and l<hi]
    if not matching:continue
    step=matching[0];features.append(hidden[prompt_len-1+step]);records.append(dict(image=path,image_id=row['image_id'],split=split,word=word,category=node,token_step=step,consistent=int(node in e.imid_to_objects[row['image_id']])))
   print('CAPTION_FEATURES',split,i+1,len(paths),'mentions',len(records),'seconds',round(time.time()-start),flush=True)
del model;torch.cuda.empty_cache()
x=torch.stack(features);y=torch.tensor([r['consistent'] for r in records]);tr=[i for i,r in enumerate(records) if r['split']=='train'];va=[i for i,r in enumerate(records) if r['split']=='validation']
assert len(y[tr].unique())==2 and len(y[va].unique())==2
write_json(a.out/'records.json',records);write_json(a.out/'skipped.json',skipped);torch.save(dict(hidden=x,labels=y,records=records),a.out/'features.pt')
v,h=train_scalar(x[tr],y[tr],x[va],y[va],a.out/'direction.pt');write_json(a.out/'training_history.json',h)
write_json(a.out/'protocol.json',dict(train_images=train,validation_images=validation,train_mentions=len(tr),validation_mentions=len(va),train_inconsistent=int((y[tr]==0).sum()),validation_inconsistent=int((y[va]==0).sum()),max_new_tokens=128,layer=24,feature_location='last decoder position before first subtoken of object mention',test_access=False,elapsed_seconds=time.time()-start,source_sha256=sha(Path(__file__))))
print('CAPTION_DIRECTION_READY',flush=True)
