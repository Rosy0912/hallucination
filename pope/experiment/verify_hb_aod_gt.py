"""Read every test image anew and verify the frozen AOD GT ensemble."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import run as core
from hb_aod_gt import AOD
from hb_aod_staged_decode import load_backbone
from aod.vlm.intervention import resolve_decoder_layers


def main():
    p=argparse.ArgumentParser()
    for name in ['source','experiment','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=['qwen','llava','internvl'],required=True)
    a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    selection=json.loads((a.experiment/'selection.json').read_text())
    assert not selection['fallback']
    cache=torch.load(a.source/'features.pt',weights_only=False)
    assert hashlib.sha256((a.source/'features.pt').read_bytes()).hexdigest()==selection['source_sha256']
    rows=cache['records'];te=[i for i,r in enumerate(rows) if r['split']=='test']
    paired='ablation_margin' in rows[0]
    if paired:
        feature_protocol=json.loads((a.source/'extraction_protocol.json').read_text())
        assert a.model=='qwen' and feature_protocol['paired_representation']=='concatenate(real hidden, real hidden - zero-pixel hidden)'
    old={r['id']:r['pred'] for r in json.loads((a.experiment/'test_predictions.json').read_text())}
    heads=[]
    for path in selection['checkpoints']:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==selection['checkpoint_sha256'][path]
        payload=torch.load(path,weights_only=False)
        net=AOD(payload['dimension'],payload.get('width',256)).cuda().eval();net.load_state_dict(payload['state_dict'])
        heads.append((net,payload))
    model,inputs_for,forward,yes,no,ties_yes=load_backbone(a.model)
    layers,_=resolve_decoder_layers(model);captured={};handles=[]
    for layer in set(payload['layer'] for _,payload in heads):
        def hook(_m,_i,out,l=layer):captured[l]=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float()
        handles.append(layers[layer-1].register_forward_hook(hook))
    predictions=[];feature_parity=True;base_parity=True;selected_parity=True
    with torch.inference_mode(),(a.out/'predictions.jsonl').open('w') as stream:
        for count,i in enumerate(te,1):
            row=rows[i];inputs=inputs_for(row);logits=forward(inputs).float();margin=float(logits[0,yes].max()-logits[0,no].max())
            bp=int(margin>=0) if ties_yes else int(margin>0);base_parity &= bp==row['pred'];values=[]
            states={layer:state.clone() for layer,state in captured.items()}
            if paired:
                masked=dict(inputs);masked['pixel_values']=torch.zeros_like(inputs['pixel_values']);forward(masked)
                states={layer:torch.cat([state,state-captured[layer]]) for layer,state in states.items()}
            for net,payload in heads:
                x=states[payload['layer']]
                feature_parity &= torch.equal(x.cpu(),cache['hidden'][payload['layer']][i].float())
                if 'mean' in payload:x=(x-payload['mean'].cuda())/payload['scale'].cuda()
                else:x=x/payload['rms'].cuda()
                if payload.get('normalization')=='standard':x=x.clamp(-10,10)
                values.append(float(net(x[None],selection['alpha'])[0]))
            score=selection['blend']*(sum(values)/len(values)+selection['recipe']['offset']*margin)+(1-selection['blend'])*margin
            pp=int(score>=selection['threshold']);selected_parity &= pp==old[row['id']]
            record=dict(row,base_pred=bp,pred=pp,score=score);predictions.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
            if count%50==0:print('VERIFIED',count,len(te),flush=True)
    for h in handles:h.remove()
    b=core.metrics([dict(r,pred=r['base_pred']) for r in predictions]);m=core.metrics(predictions)
    result=dict(method='AOD',model=a.model,base=b,selected=m,accuracy_gain_pp=m['accuracy']-b['accuracy'],
                base_predictions_exact=base_parity,features_exact=feature_parity,selected_predictions_exact=selected_parity,
                complete_test=len(predictions)==len(te),criterion_met=m['accuracy']-b['accuracy']>=5,paired_visual_features=paired,
                selection=selection)
    core.atomic_json(a.out/'metrics.json',result);print('RESULT',json.dumps(result),flush=True)
    assert base_parity and feature_parity and selected_parity


if __name__=='__main__':main()
