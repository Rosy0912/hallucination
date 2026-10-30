"""Shared, frozen LLaVA inputs and AOD training for the new benchmark runs."""
import hashlib
import json
import os
import random
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def binary_metrics(rows, key='pred'):
    if not rows:
        raise ValueError('Cannot score an empty set')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prediction IDs')
    if any(r['gt'] not in (0, 1) or r[key] not in (0, 1) for r in rows):
        raise ValueError('Binary labels must be 0 or 1')
    tp = sum(r['gt'] == 1 and r[key] == 1 for r in rows)
    fp = sum(r['gt'] == 0 and r[key] == 1 for r in rows)
    tn = sum(r['gt'] == 0 and r[key] == 0 for r in rows)
    fn = len(rows) - tp - fp - tn
    return dict(n=len(rows), tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=100*(tp+tn)/len(rows), precision=100*tp/max(1,tp+fp),
                recall=100*tp/max(1,tp+fn), f1=200*tp/max(1,2*tp+fp+fn),
                false_positive_rate=100*fp/max(1,fp+tn))


def group_split(records, limit=200, seed=42):
    images = sorted({r['image'] for r in records})
    random.Random(seed).shuffle(images)
    if limit:
        images = images[:limit]
    if len(images) < 5:
        raise ValueError('At least five images are required')
    cut = int(len(images)*0.8)
    train, test = set(images[:cut]), set(images[cut:])
    return [dict(r, split='train' if r['image'] in train else 'test')
            for r in records if r['image'] in train | test]


def load_llava(model_path):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('GPU inference requires a CUDA allocation')
    torch.set_num_threads(4)
    proc = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='cuda:0').eval()
    proc.patch_size = model.config.vision_config.patch_size
    proc.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    proc.num_additional_image_tokens = 1
    return model, proc


def inputs_for(proc, image_path, question):
    import torch
    from PIL import Image
    with Image.open(image_path) as im:
        x = proc(text='USER: <image>\n'+question+' ASSISTANT:',
                 images=im.convert('RGB'), return_tensors='pt')
    return {k: v.to('cuda:0', dtype=torch.float16 if v.is_floating_point() else v.dtype)
            for k, v in x.items()}


def train_direction(x, y, layer, output, seed=42, epochs=5):
    import torch
    from aod.core.aod import AODCheckpoint, AODDisentangler, save_checkpoint
    torch.manual_seed(seed)
    x, y = x.float().cuda(), y.float().cuda().view(-1, 1)
    if len(set(y.view(-1).cpu().tolist())) != 2:
        raise ValueError('Training consistency labels must include both classes')
    net = AODDisentangler(x.shape[1], 512, 1.0).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(epochs):
        total = 0.
        for idx in torch.randperm(len(x), device=x.device).split(256):
            z = net(x[idx])
            loss = loss_fn(z['pred_consistency'], y[idx]) + loss_fn(z['pred_residual_adv'], y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite AOD training loss')
            opt.zero_grad(); loss.backward(); opt.step(); net.renormalize_direction_()
            total += float(loss.detach())*len(idx)
        losses.append(total/len(x))
        print('TRAIN', epoch+1, losses[-1], flush=True)
    meta = AODCheckpoint(layer, x.shape[1], 512, 1.0, seed)
    save_checkpoint(str(output), meta, net.cpu().eval())
    return net.v_unit().detach().view(-1), losses


def provenance():
    import torch
    import transformers
    return dict(host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                torch=torch.__version__, transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0),
                source_sha256={str(p.relative_to(ROOT)): sha(p) for folder in
                               ['src', 'tools', 'amber', 'chair'] for p in (ROOT/folder).rglob('*.py')})
