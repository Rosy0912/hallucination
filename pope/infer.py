"""Evaluate a freshly trained POPE head using fresh image forwards."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiment'))
import torch
from run import metrics,atomic_json
from user_gt_models import GTModel
from hb_aod_staged_decode import load_backbone
from aod.vlm.intervention import resolve_decoder_layers

def main():
    p=argparse.ArgumentParser()
    for name in ['source','trained','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision("high")
    if a.out.exists():raise FileExistsError(a.out)
    cache=torch.load(a.source/'features.pt',map_location='cpu',weights_only=False)
    selection=json.loads((a.trained/'selection.json').read_text())['projective']
    payload=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)
    assert payload['source_sha256']==hashlib.sha256((a.source/'features.pt').read_bytes()).hexdigest()
    net=GTModel(4096,'projective').cuda().eval();net.load_state_dict(payload['state_dict'])
    model,inputs_for,forward,yes,no,ties=load_backbone('llava')
    layers,_=resolve_decoder_layers(model);captured={}
    def hook(_m,_i,out):captured['x']=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float()
    handle=layers[selection['layer']-1].register_forward_hook(hook)
    a.out.mkdir(parents=True);rows=[]
    with torch.inference_mode(),(a.out/'predictions.jsonl').open('w') as f:
        for r in cache['records']:
            if r['split']!='test':continue
            logits=forward(inputs_for(r)).float()[0]
            base=int(logits[yes].max()>logits[no].max())
            score=float(net(captured['x'][None],selection['alpha'])[0].item())
            row=dict(r,base_pred=base,pred=int(score>=selection['threshold']),score=score)
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
    handle.remove()
    result=dict(base=metrics([dict(r,pred=r['base_pred']) for r in rows]),aod=metrics(rows),selection=selection)
    atomic_json(a.out/'metrics.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
