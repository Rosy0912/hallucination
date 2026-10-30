"""AOD orthogonal disentanglement with explicit optimization diagnostics.

Keep raw-origin rank-one geometry: no centering, no answer classifier, no
threshold correction. One positive scalar rescales training features for
conditioning; inference uses the original raw projection (z.v)v unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import torch
from run import atomic_json
from aod.core.aod import AODDisentangler,AODCheckpoint,save_checkpoint


def diagnostic(net,x,y):
    with torch.no_grad():
        output=net(x);target=y.view(-1).cpu().numpy()
        result={}
        for name,field in [('projection','pred_consistency'),('residual','pred_residual_adv')]:
            logits=output[field].view(-1).cpu().numpy()
            result[name+'_balanced_accuracy']=100*balanced_accuracy_score(target,logits>=0)
            result[name+'_auroc']=100*roc_auc_score(target,logits)
        v=net.v_unit();h=output['h_hallu'];r=output['h_res']
        result['direction_norm']=float(v.norm())
        result['relative_orthogonality_error']=float(((h*r).sum(-1).abs()/(h.norm(dim=-1)*r.norm(dim=-1)).clamp_min(1e-12)).max())
        result['projection_energy_ratio']=float(h.square().sum()/x.square().sum())
        result['predicted_consistent_fraction']=float((output['pred_consistency']>=0).float().mean())
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--layers',default='20,24,28')
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--adv-weights',default='0.1,1.0')
    ap.add_argument('--fixed-epoch',action='store_true',help='Save the declared final epoch, without validation epoch selection')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);(args.out/'checkpoints').mkdir(exist_ok=True)
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    cache=torch.load(args.source/'features.pt',weights_only=False);rows=cache['records']
    assert len({r['benchmark'] for r in rows})==1
    benchmark=rows[0]['benchmark'];train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='val']
    assert not({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    y=torch.tensor([int(r['pred']==r['gt']) for r in rows],dtype=torch.float32,device=device)[:,None]
    source_sha=hashlib.sha256((args.source/'features.pt').read_bytes()).hexdigest();ranking=[]
    for layer in [int(l) for l in args.layers.split(',')]:
        raw=torch.stack(cache['hidden'][layer]).float().to(device)
        scale=raw[train].square().mean().sqrt().clamp_min(1e-6)
        x=raw/scale
        for adv_weight in map(float,args.adv_weights.split(',')):
            tag=f'{benchmark}_L{layer}_orthogonal_scaled_balanced_adv{adv_weight}_s{args.seed}'
            path=args.out/'checkpoints'/f'{tag}.pt'
            if path.exists():
                ranking.append(json.loads(path.with_suffix('.json').read_text()));continue
            torch.manual_seed(args.seed);np.random.seed(args.seed)
            net=AODDisentangler(x.shape[1],512,adv_weight).to(device)
            # Mean-difference initialization retains original AOD projection geometry.
            difference=x[train][y[train].view(-1)==1].mean(0)-x[train][y[train].view(-1)==0].mean(0)
            with torch.no_grad():net.v.copy_(difference[None]);net.renormalize_direction_()
            initial=net.v_unit().detach().clone()
            n_positive=y[train].sum();weights=torch.where(y[train]>0,len(train)/(2*n_positive),len(train)/(2*(len(train)-n_positive)))
            opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=0)
            logs=[];best=None;state=None;selected_epoch=None
            for epoch in range(args.epochs):
                net.train();loss_sum=0.;grad_norm=0.
                for index in torch.randperm(len(train),device=device).split(256):
                    batch=torch.tensor(train,device=device)[index];output=net(x[batch])
                    cls=torch.nn.functional.binary_cross_entropy_with_logits(output['pred_consistency'],y[batch],reduction='none')
                    adv=torch.nn.functional.binary_cross_entropy_with_logits(output['pred_residual_adv'],y[batch],reduction='none')
                    loss=((cls+adv_weight*adv)*weights[index]).mean()
                    opt.zero_grad();loss.backward();grad_norm=float(net.v.grad.norm())
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    opt.step();net.renormalize_direction_();loss_sum+=float(loss)*len(index)
                net.eval();diagnostics=diagnostic(net,x[val],y[val])
                rec=dict(epoch=epoch+1,loss=loss_sum/len(train),direction_gradient_norm=grad_norm,**diagnostics);logs.append(rec)
                # A probe selects a usable candidate; downstream decoding selection is separate.
                key=(rec['projection_balanced_accuracy'],rec['projection_auroc'])
                if args.fixed_epoch or best is None or key>best:
                    best=key;selected_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                if (epoch+1)%10==0:print(tag,rec,flush=True)
            net.load_state_dict(state);net.eval()
            final=diagnostic(net,x[val],y[val]);final['direction_change_l2']=float((net.v_unit()-initial).norm())
            assert final['relative_orthogonality_error']<1e-4
            net=net.cpu();save_checkpoint(str(path),AODCheckpoint(layer,x.shape[1],512,adv_weight,args.seed),net)
            saved=torch.load(path,weights_only=False);saved['training_scale']=scale.cpu();torch.save(saved,path)
            provenance=dict(tag=tag,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256=source_sha,benchmark=benchmark,layer=layer,adv_weight=adv_weight,
                variant='AOD raw-origin orthogonal decomposition, globally scaled training, balanced BCE, mean-difference initialization, lr 1e-4, gradient clip 1',
                center=None,train_ids=[rows[i]['id'] for i in train],validation_ids=[rows[i]['id'] for i in val],
                training_n=len(train),validation_n=len(val),epochs=logs,selected_epoch=selected_epoch,validation_diagnostics=final,
                checkpoint_rule='Declared final epoch' if args.fixed_epoch else 'Validation probe ranking')
            atomic_json(path.with_suffix('.json'),provenance);ranking.append(provenance)
            print('SAVED',tag,final,flush=True)
    ranking.sort(key=lambda r:(-r['validation_diagnostics']['projection_balanced_accuracy'],-r['validation_diagnostics']['projection_auroc'],r['layer'],r['adv_weight']))
    selected=[];used_layers=set()
    for row in ranking:
        if row['layer'] in used_layers:continue
        selected.append(row);used_layers.add(row['layer'])
        if len(selected)==2:break
    atomic_json(args.out/'probe_ranking.json',ranking)
    atomic_json(args.out/'decoding_candidates.json',dict(rule='Top two distinct layers by validation projection balanced accuracy then AUROC; no test access',
        candidates=[{k:r[k] for k in ['checkpoint','checkpoint_sha256','layer','adv_weight','selected_epoch','validation_diagnostics']} for r in selected]))
    print('DECODING CANDIDATES',[r['checkpoint'] for r in selected],flush=True)


if __name__=='__main__':main()
