"""HallusionBench AOD: GT-supervised orthogonal residual and paired training.

Only train labels update parameters. Validation chooses checkpoints, layers,
alpha and blending. No test features are forwarded until selection is saved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from run import atomic_json, metrics
from aod.core.aod import GradientReversalLayer


class AOD(nn.Module):
    def __init__(self, dimension, width=256):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dimension))
        self.truth = nn.Sequential(nn.Linear(dimension, width), nn.LayerNorm(width),
                                   nn.ReLU(), nn.Dropout(.15), nn.Linear(width, 1))
        self.adversary = nn.Sequential(GradientReversalLayer(), nn.Linear(dimension, 64),
                                      nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, alpha=1.):
        v = nn.functional.normalize(self.v, dim=0)
        projection = (x @ v)[:, None] * v
        return self.truth(x - alpha * projection).flatten(), self.adversary(projection).flatten()


def pairs_in_training(rows, train):
    groups = {}
    for local, i in enumerate(train):
        r = rows[i]
        key = (r['group'], str(r['raw']['question_id']))
        groups.setdefault(key, {0: [], 1: []})[r['gt']].append(local)
    return [(a, b) for g in groups.values() for a in g[1] for b in g[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', required=True)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--normalization', choices=['rms', 'standard'], default='rms')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--balanced', action='store_true')
    p.add_argument('--f1-guard', action='store_true')
    a = p.parse_args()
    assert not a.out.exists(), 'Use a fresh output directory'
    a.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = torch.load(a.source / 'features.pt', weights_only=False)
    rows = cache['records']
    ids = {s: np.array([i for i, r in enumerate(rows) if r['split'] == s]) for s in ['train', 'val', 'test']}
    tr, va, te = (ids[s] for s in ['train', 'val', 'test'])
    for s, t in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not {rows[i]['group'] for i in ids[s]} & {rows[i]['group'] for i in ids[t]}
    assert all(r['image'] for r in rows)
    y = torch.tensor([rows[i]['gt'] for i in tr], dtype=torch.float32, device=device)
    yv = torch.tensor([rows[i]['gt'] for i in va], dtype=torch.float32, device=device)
    weights = torch.where(y>0, len(tr)/(2*y.sum()), len(tr)/(2*(len(tr)-y.sum()))) if a.balanced else torch.ones_like(y)
    mv = np.array([rows[i]['margin'] for i in va])
    mt = torch.tensor([rows[i]['margin'] for i in tr], dtype=torch.float32, device=device)
    pair_list = pairs_in_training(rows, tr)
    assert pair_list, 'Pair ranking requires matched opposite-label training examples'
    pairs = torch.tensor(pair_list, device=device)
    layers = list(map(int, a.layers.split(',')))
    seeds = list(map(int, a.seeds.split(',')))
    alphas = [-1., -.5, 0., .5, 1.]
    source_sha = hashlib.sha256((a.source / 'features.pt').read_bytes()).hexdigest()
    protocol = dict(method='AOD', training_target='GT Yes/No', token='last prompt token',
                    source_sha256=source_sha, layers=layers, seeds=seeds, alphas=alphas,
                    normalization=a.normalization, width=a.width, balanced=a.balanced, f1_guard=a.f1_guard,
                    pair_count=len(pair_list), pairs='Same group and question, opposite GT on different figures',
                    training_ids=[rows[i]['id'] for i in tr], validation_ids=[rows[i]['id'] for i in va],
                    test_ids=[rows[i]['id'] for i in te], selection='Validation only; original grouped split unchanged')
    atomic_json(a.out / 'protocol.json', protocol)
    candidates = []
    checkpoint_logs = []
    # Two fixed training objectives; all are AOD, with GT-supervised adversary.
    recipes = [dict(name='paired', offset=0., pair_weight=.5, penalty=0.),
               dict(name='paired_residual', offset=1., pair_weight=.5, penalty=.01)]
    for layer in layers:
        raw_train = torch.stack([cache['hidden'][layer][i] for i in tr]).float().to(device)
        raw_val = torch.stack([cache['hidden'][layer][i] for i in va]).float().to(device)
        # Global scaling preserves angles and the raw-origin projection.
        rms = raw_train.square().mean().sqrt().clamp_min(1e-6)
        mean = raw_train.mean(0) if a.normalization=='standard' else torch.zeros(raw_train.shape[1],device=device)
        scale = raw_train.std(0).clamp_min(.01) if a.normalization=='standard' else rms
        xt, xv = (raw_train-mean)/scale, (raw_val-mean)/scale
        if a.normalization=='standard':
            xt, xv = xt.clamp(-10,10), xv.clamp(-10,10)
        for recipe in recipes:
            paths, val_seed_scores = [], []
            for seed in seeds:
                torch.manual_seed(seed)
                model = AOD(xt.shape[1],a.width).to(device)
                if recipe['offset']:
                    nn.init.zeros_(model.truth[-1].weight)
                    nn.init.zeros_(model.truth[-1].bias)
                opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.05)
                best_loss, best_state, best_epoch = float('inf'), None, None
                logs = []
                for ep in range(1, a.epochs + 1):
                    model.train()
                    delta, adv = model(xt + .01 * torch.randn_like(xt))
                    score = delta + recipe['offset'] * mt
                    main_loss = (nn.functional.binary_cross_entropy_with_logits(score, .98 * y + .01,reduction='none')*weights).mean()
                    pair_loss = nn.functional.softplus(-(score[pairs[:, 0]] - score[pairs[:, 1]])).mean()
                    loss = main_loss + .1 * nn.functional.binary_cross_entropy_with_logits(adv, y)
                    loss = loss + recipe['pair_weight'] * pair_loss + recipe['penalty'] * delta.square().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    if ep % 5:
                        continue
                    model.eval()
                    with torch.no_grad():
                        vd = model(xv)[0]
                        vs = vd + recipe['offset'] * torch.as_tensor(mv, device=device)
                        vl = float(nn.functional.binary_cross_entropy_with_logits(vs, yv))
                    logs.append(dict(epoch=ep, validation_bce=vl, train_loss=float(loss.detach())))
                    if vl < best_loss:
                        best_loss, best_epoch = vl, ep
                        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    val_seed_scores.append(np.stack([model(xv, alpha)[0].cpu().numpy() for alpha in alphas]))
                ck = a.out / f'L{layer}_{recipe["name"]}_s{seed}.pt'
                torch.save(dict(state_dict=best_state, dimension=xt.shape[1], width=a.width, rms=rms.cpu(),
                                mean=mean.cpu(), scale=scale.cpu(), normalization=a.normalization, recipe=recipe,
                                layer=layer, seed=seed, epoch=best_epoch, source_sha256=source_sha,
                                train_ids=protocol['training_ids']), ck)
                paths.append(str(ck))
                checkpoint_logs.append(dict(checkpoint=str(ck), best_epoch=best_epoch,
                                            validation_bce=best_loss, history=logs))
                del model, opt
            scores = np.mean(val_seed_scores, axis=0)
            candidates.append(dict(layers=[layer], recipe=recipe, checkpoints=paths, scores=scores))
            print('TRAINED', layer, recipe['name'], 'seeds', seeds, flush=True)
        del raw_train, raw_val, xt, xv
    # Predefined layer ensembles, not selected using test predictions.
    for recipe in recipes:
        components = [c for c in candidates if c['recipe'] == recipe]
        for chosen in [components, components[len(components)//2:]]:
            if len(chosen) > 1:
                candidates.append(dict(layers=[l for c in chosen for l in c['layers']], recipe=recipe,
                                       checkpoints=[p for c in chosen for p in c['checkpoints']],
                                       scores=np.mean([c['scores'] for c in chosen], axis=0)))
    base_val = metrics([rows[i] for i in va])
    base_test = metrics([rows[i] for i in te])
    yy = yv.cpu().numpy().astype(bool)
    search = []
    best = dict(fallback=True, validation=base_val)
    best_key = (base_val['accuracy'], base_val['f1'], 0.)
    for c in candidates:
        for ai, alpha in enumerate(alphas):
            for blend in [.25, .5, 1.]:
                # Both inputs are logit-scale Yes/No scores.
                score = blend * (c['scores'][ai] + c['recipe']['offset'] * mv) + (1 - blend) * mv
                for threshold in [-.5, 0., .5]:
                    pred = score >= threshold
                    tp, tn = int(np.sum(pred & yy)), int(np.sum(~pred & ~yy))
                    fp, fn = int(np.sum(pred & ~yy)), int(np.sum(~pred & yy))
                    m = dict(n=len(va), accuracy=100*(tp+tn)/len(va), f1=200*tp/max(2*tp+fp+fn, 1))
                    cfg = {k: v for k, v in c.items() if k != 'scores'}
                    cfg.update(fallback=False, alpha=alpha, blend=blend, threshold=threshold, validation=m)
                    search.append(cfg)
                    if a.f1_guard and m['f1'] < base_val['f1']:
                        continue
                    key = (m['accuracy'], m['f1'], -abs(threshold))
                    if key > best_key:
                        best, best_key = cfg, key
    best.update(method='AOD', source_sha256=source_sha, alpha_grid=alphas,
                f1_guard=a.f1_guard, rule='Validation accuracy then F1 with configured F1 guard; checkpoints chosen by validation BCE; Base fallback eligible')
    if not best['fallback']:
        best['checkpoint_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in best['checkpoints']}
    atomic_json(a.out / 'selection.json', best)
    atomic_json(a.out / 'validation_search.json', search)
    atomic_json(a.out / 'training_history.json', checkpoint_logs)
    # Test inference begins only after frozen selection is persisted.
    if best['fallback']:
        pp = np.array([rows[i]['pred'] for i in te])
    else:
        outputs = []
        for path in best['checkpoints']:
            payload = torch.load(path, weights_only=False)
            x = torch.stack([cache['hidden'][payload['layer']][i] for i in te]).float().to(device)
            x = (x-payload['mean'].to(device))/payload['scale'].to(device)
            if payload['normalization']=='standard':x=x.clamp(-10,10)
            model = AOD(payload['dimension'],payload['width']).to(device)
            model.load_state_dict(payload['state_dict'])
            model.eval()
            with torch.no_grad():
                outputs.append(model(x, best['alpha'])[0].cpu().numpy())
        margin = np.array([rows[i]['margin'] for i in te])
        score = best['blend'] * (np.mean(outputs, axis=0) + best['recipe']['offset'] * margin) + (1-best['blend']) * margin
        pp = score >= best['threshold']
    pred = [dict(rows[i], base_pred=rows[i]['pred'], pred=int(v)) for i, v in zip(te, pp)]
    selected = metrics(pred)
    result = dict(method='AOD', base=base_test, selected=selected, selection=best,
                  accuracy_gain_pp=selected['accuracy']-base_test['accuracy'],
                  corrected=sum(r['base_pred'] != r['gt'] and r['pred'] == r['gt'] for r in pred),
                  regressed=sum(r['base_pred'] == r['gt'] and r['pred'] != r['gt'] for r in pred),
                  criterion_met=selected['accuracy']-base_test['accuracy'] >= 5)
    atomic_json(a.out / 'test_predictions.json', pred)
    atomic_json(a.out / 'results.json', result)
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
