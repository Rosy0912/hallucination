"""AOD normalized direction training followed by actual internal intervention."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from PIL import Image
import run as core
from aod_staged_direction import StagedDirection
from aod.vlm.intervention import layer_steering_hook
from aod.vlm.answers import llava_answer_ids


def load_backbone(name):
    if name in ['qwen','internvl']:
        directory='Qwen2.5-VL-7B-Instruct' if name=='qwen' else 'InternVL3-8B-hf'
        loaded = core.load_model(SimpleNamespace(model=str(core.ROOT/'models'/directory), max_pixels=1003520 if name=='qwen' else None))
        yes, no = core.get_yes_no_token_ids(loaded.tokenizer)
        return loaded.model, lambda r: core.inputs_for(loaded, r), lambda x: core.forward_next_logits(loaded.model, x), yes, no, True
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    path = core.ROOT/'models/llava-1.5-7b-hf'
    proc = AutoProcessor.from_pretrained(path, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16, device_map='cuda:0').eval()
    proc.patch_size = model.config.vision_config.patch_size
    proc.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    proc.num_additional_image_tokens = 1
    def inputs(r):
        with Image.open(core.ROOT/'data'/r['image']) as image:
            x = proc(text='USER: <image>\n'+r['question']+' ASSISTANT:', images=image.convert('RGB'), return_tensors='pt')
        return {k: v.to('cuda:0', dtype=torch.float16 if v.is_floating_point() else v.dtype) for k, v in x.items()}
    yes_id,no_id=llava_answer_ids(proc.tokenizer)
    yes,no=[yes_id],[no_id]
    return model, inputs, lambda x: model(**x, use_cache=False).logits[:, -1, :], yes, no, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['qwen', 'llava'], required=True)
    p.add_argument('--layers', default='16,20,24,28')
    p.add_argument('--top-layers', type=int, default=2)
    a = p.parse_args()
    assert not a.out.exists()
    a.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    cache = torch.load(a.source/'features.pt', weights_only=False)
    rows = cache['records']
    ids = {s: [i for i,r in enumerate(rows) if r['split']==s] for s in ['train','val','test']}
    tr, va, te = (ids[s] for s in ['train','val','test'])
    assert all(r['image'] for r in rows)
    for s,t in [('train','val'),('train','test'),('val','test')]:
        assert not {rows[i]['group'] for i in ids[s]} & {rows[i]['group'] for i in ids[t]}
    correct = np.array([int(r['pred']==r['gt']) for r in rows])
    cy = torch.tensor(correct[tr], dtype=torch.float32, device='cuda')
    weights = torch.where(cy>0, len(tr)/(2*cy.sum()), len(tr)/(2*(len(tr)-cy.sum())))
    directions, diagnostics = {}, []
    for layer in map(int, a.layers.split(',')):
        xt = torch.stack([cache['hidden'][layer][i] for i in tr]).float().cuda()
        xv = torch.stack([cache['hidden'][layer][i] for i in va]).float().cuda()
        rms = xt.square().mean().sqrt().clamp_min(1e-6)
        xt, xv = xt/rms, xv/rms
        torch.manual_seed(42)
        net = StagedDirection(xt, cy).cuda()
        opt = torch.optim.AdamW([{'params':[net.v], 'lr':5e-4},
                                {'params':net.consistency.parameters(), 'lr':5e-3},
                                {'params':net.adversary.parameters(), 'lr':5e-4}], weight_decay=.01)
        for epoch in range(100):
            net.train()
            for ib in torch.randperm(len(tr), device='cuda').split(128):
                cp, adv = net(xt[ib])
                loss = ((F.binary_cross_entropy_with_logits(cp, cy[ib], reduction='none') +
                         .1*F.binary_cross_entropy_with_logits(adv, cy[ib], reduction='none'))*weights[ib]).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.)
                opt.step()
        net.eval()
        with torch.no_grad():
            auc = float(roc_auc_score(correct[va], net(xv)[0].cpu().numpy()))
            train_auc = float(roc_auc_score(correct[tr], net(xt)[0].cpu().numpy()))
        directions[layer] = F.normalize(net.v, dim=1).flatten().detach().cpu()
        checkpoint = a.out/f'direction_L{layer}.pt'
        torch.save(dict(state_dict=net.cpu().state_dict(), direction=directions[layer], layer=layer,
                        rms=rms.cpu(), train_ids=[rows[i]['id'] for i in tr]), checkpoint)
        diagnostics.append(dict(layer=layer, checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                train_auroc=train_auc, validation_auroc=auc))
        print('DIRECTION', diagnostics[-1], flush=True)
        del net, opt, xt, xv
    selected_layers = [r['layer'] for r in sorted(diagnostics, key=lambda d:d['validation_auroc'], reverse=True)[:a.top_layers]]
    source_sha = hashlib.sha256((a.source/'features.pt').read_bytes()).hexdigest()
    protocol = dict(method='AOD', model=a.model, source_sha256=source_sha, direction_epochs=100,
                    selected_layers=selected_layers, layer_selection='Top validation consistency AUROC',
                    alphas=[-1.,-.5,0.,.5,1.], betas=[0.,.5,1.,2.], apc='none',
                    token='Last prompt token', training_ids=[rows[i]['id'] for i in tr])
    core.atomic_json(a.out/'protocol.json', protocol)
    core.atomic_json(a.out/'direction_diagnostics.json', diagnostics)
    torch.cuda.empty_cache()
    model, inputs_for, forward, yes, no, ties_yes = load_backbone(a.model)
    label_ids = yes + no
    def pred(logits):
        margin = float(logits[:len(yes)].max() - logits[len(yes):].max())
        return int(margin>=0) if ties_yes else int(margin>0)
    configs = [dict(layer=l, alpha=alpha, beta=beta) for l in selected_layers
               for alpha in protocol['alphas'] for beta in protocol['betas']]
    val_predictions = [[] for _ in configs]
    with torch.inference_mode(), (a.out/'validation_logits.jsonl').open('w') as stream:
        for count,i in enumerate(va, 1):
            row = rows[i]
            x = inputs_for(row)
            base = forward(x).float()[0,label_ids].cpu()
            assert pred(base)==row['pred'], 'Actual image Base prediction mismatch'
            grid = {}
            for layer in selected_layers:
                for strength in [-1.,-.5,.5,1.]:
                    with layer_steering_hook(model, layer, directions[layer], strength):
                        grid[layer,strength] = forward(x).float()[0,label_ids].cpu()
            for j,cfg in enumerate(configs):
                alpha, beta, layer = cfg['alpha'],cfg['beta'],cfg['layer']
                logits = base if alpha==0 else (1+beta)*grid[layer,-alpha]-beta*grid[layer,alpha]
                val_predictions[j].append(dict(row,pred=pred(logits)))
            stream.write(json.dumps(dict(id=row['id'], gt=row['gt'], base=base.tolist(),
                                         logits={f'{k[0]}/{k[1]}':v.tolist() for k,v in grid.items()}))+'\n')
            stream.flush()
            if count%25==0: print('VALIDATION',count,len(va),flush=True)
    bv = core.metrics([rows[i] for i in va])
    best, best_key = dict(fallback=True, validation=bv), (bv['accuracy'],bv['f1'],0.,0.)
    ranking = []
    for cfg, predictions in zip(configs, val_predictions):
        m = core.metrics(predictions)
        entry = dict(cfg, validation=m)
        ranking.append(entry)
        if m['f1'] < bv['f1']:
            continue
        key = (m['accuracy'],m['f1'],-abs(cfg['alpha']),-cfg['beta'])
        if key>best_key:
            best_key, best = key, dict(entry,fallback=False)
    best.update(method='AOD', source_sha256=source_sha,
                rule='Validation accuracy with F1 >= Base, then F1; test after freeze')
    if not best['fallback']:
        best['direction'] = next(r for r in diagnostics if r['layer']==best['layer'])
    core.atomic_json(a.out/'selection.json', best)
    core.atomic_json(a.out/'validation_ranking.json', ranking)
    test_predictions=[]
    with torch.inference_mode(), (a.out/'test_predictions.jsonl').open('w') as stream:
        for count,i in enumerate(te,1):
            row=rows[i]
            x=inputs_for(row)
            base=forward(x).float()[0,label_ids].cpu()
            bp=pred(base)
            assert bp==row['pred'], 'Test Base mismatch'
            logits=base
            if not best['fallback'] and best['alpha']:
                layer,alpha,beta=best['layer'],best['alpha'],best['beta']
                with layer_steering_hook(model,layer,directions[layer],-alpha):pos=forward(x).float()[0,label_ids].cpu()
                with layer_steering_hook(model,layer,directions[layer],alpha):neg=forward(x).float()[0,label_ids].cpu()
                logits=(1+beta)*pos-beta*neg
            record=dict(row,base_pred=bp,pred=pred(logits),base_logits=base.tolist(),aod_logits=logits.tolist())
            test_predictions.append(record)
            stream.write(json.dumps(record)+'\n');stream.flush()
            if count%50==0: print('TEST',count,len(te),flush=True)
    b=core.metrics([rows[i] for i in te]);m=core.metrics(test_predictions)
    result=dict(method='AOD',base=b,selected=m,selection=best,accuracy_gain_pp=m['accuracy']-b['accuracy'],
                corrected=sum(r['base_pred']!=r['gt'] and r['pred']==r['gt'] for r in test_predictions),
                regressed=sum(r['base_pred']==r['gt'] and r['pred']!=r['gt'] for r in test_predictions),
                criterion_met=m['accuracy']-b['accuracy']>=5,base_predictions_exact=True,actual_images=True)
    core.atomic_json(a.out/'test_predictions.json',test_predictions)
    core.atomic_json(a.out/'results.json',result)
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
