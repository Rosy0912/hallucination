"""Caption-domain AOD: training, disjoint validation, frozen evaluation.

No precomputed results, trained checkpoints or server-specific paths are needed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.runtime import write_json, sha

PROMPT = 'Please describe this image in detail.'
CONFIGS = [dict(name='projection_'+str(x), mode='projection_cd', alpha=x, beta=.5)
           for x in [.5, 1., 2., -1.]] + [
    dict(name='additive_'+str(x), mode='additive', alpha=x, beta=0)
    for x in [.5, 1., 2., -1.]]


def passes(base, candidate):
    return (all(base[k] is not None and candidate[k] is not None and candidate[k] < base[k]
                for k in ['CHAIRs', 'CHAIRi'])
            and all(candidate[k] >= .9 * base[k] for k in
                    ['mean_tokens', 'correct_unique_objects_per_image', 'object_recall']))


def quality(evaluator, rows, name):
    from evaluate import score
    result = score(evaluator, rows, name+'_caption')
    sentences = result['sentences']
    result['mean_tokens'] = sum(r[name+'_tokens'] for r in rows) / len(rows)
    correct = [len(set(s['generated_objects']) & set(s['ground_truth'])) for s in sentences]
    result['correct_unique_objects_per_image'] = sum(correct) / len(rows)
    result['object_recall'] = sum(n / max(1, len(s['ground_truth']))
                                  for n, s in zip(correct, sentences)) / len(rows)
    return result


def score_rows(rows, names, annotations):
    from official_algorithm import CHAIR
    evaluator = CHAIR([r['image_id'] for r in rows], str(annotations))
    evaluator.get_annotations()
    return {name: quality(evaluator, rows, name) for name in names}


def infer(paths, configs, data_root, model_path, training, output, max_new_tokens):
    import torch
    from tools.runtime import load_llava, inputs_for
    from aod.vlm.intervention import AODDecodeConfig, greedy_generate_ids, resolve_decoder_layers
    inference_config = dict(images=paths, configs=configs, tokens=max_new_tokens,
        direction_sha256=sha(training/'direction.pt'), features_sha256=sha(training/'features.pt'),
        prompt=PROMPT, script_sha256=sha(Path(__file__)))
    config_path = output.with_suffix('.protocol.json')
    if config_path.exists() and json.loads(config_path.read_text()) != inference_config:
        raise ValueError('Inference inputs changed; use a fresh --run-dir')
    if output.exists():
        if not config_path.exists():
            raise ValueError('Missing inference protocol; use a fresh --run-dir')
        rows = [json.loads(s) for s in output.read_text().splitlines() if s]
        if [r['image'] for r in rows] != paths:
            raise RuntimeError('Incomplete predictions; retain this run and choose a fresh --run-dir')
        return rows
    cache = torch.load(training/'features.pt', map_location='cpu', weights_only=False)
    ids = [i for i, r in enumerate(cache['records']) if r['split'] == 'train']
    x, y = cache['hidden'][ids].float(), cache['labels'][ids]
    checkpoint = torch.load(training/'direction.pt', map_location='cpu', weights_only=False)
    v = torch.nn.functional.normalize(checkpoint['direction'].float(), dim=0)
    if v @ (x[y > 0].mean(0)-x[y == 0].mean(0)) < 0:
        v = -v
    sigma = float((x @ v).std())
    del cache, x, y
    model, processor = load_llava(model_path)
    layers, _ = resolve_decoder_layers(model)

    @contextmanager
    def additive(alpha):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = hidden.clone()
            changed[:, -1, :] += v.to(hidden.device, hidden.dtype) * (sigma * alpha)
            return (changed,) + output[1:] if isinstance(output, tuple) else changed
        handle = layers[23].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    write_json(config_path, inference_config)
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), output.open('x') as handle:
        for i, path in enumerate(paths):
            inputs = inputs_for(processor, data_root/path, PROMPT)
            row = dict(image=path, image_id=int(Path(path).stem.rsplit('_', 1)[-1]))
            for name, cfg in [('base', None)] + [(c['name'], c) for c in configs]:
                context = additive(cfg['alpha']) if cfg and cfg['mode'] == 'additive' else nullcontext()
                decoding = AODDecodeConfig(layer=24, direction=v, alpha=cfg['alpha'],
                    beta=cfg['beta'], apc_alpha=.1, mode='cd') if cfg and cfg['mode'] == 'projection_cd' else None
                with context:
                    ids = greedy_generate_ids(model, inputs, cfg=decoding,
                        max_new_tokens=max_new_tokens, eos_token_ids=[processor.tokenizer.eos_token_id])
                row[name+'_caption'] = processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
                row[name+'_tokens'] = ids.shape[1]
            rows.append(row)
            handle.write(json.dumps(row)+'\n')
            handle.flush()
            print(output.stem, i+1, len(paths), flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['all', 'prepare', 'extract', 'train', 'validate', 'evaluate'])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--test-images', type=int, default=200)
    p.add_argument('--max-new-tokens', type=int, default=128)
    a = p.parse_args()
    if not 1 <= a.test_images <= 200:
        raise ValueError('--test-images must be in 1..200')
    if a.max_new_tokens != 128:
        raise ValueError('This protocol fixes 128 tokens for Base/AOD; use the other variant for length studies')
    os.environ['NLTK_DATA'] = str(ROOT/'data/nltk_data')
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    settings = dict(protocol='caption-domain-v1', train_images=60, search_images=16,
        confirmation_images=84, test_images=a.test_images, max_new_tokens=128,
        seed=20261031, prompt=PROMPT, configs=CONFIGS,
        model_revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
    if (run/'run_config.json').exists() and json.loads((run/'run_config.json').read_text()) != settings:
        raise ValueError('Changed protocol: use a fresh --run-dir')
    write_json(run/'run_config.json', settings)
    manifest = json.loads((ROOT/'assets/manifest.json').read_text())
    validation = sorted({r['image'] for r in manifest if r['split'] == 'val'})
    random.Random(20261030).shuffle(validation)
    assert len(validation) == 100
    annotations = ROOT/'data/coco/annotations'
    original = {int(Path(r['image']).stem.rsplit('_', 1)[-1]) for r in manifest}
    inventory = json.loads((annotations/'captions_val2014.json').read_text())['images']
    candidates = sorted([dict(image_id=r['id'], image='images/'+r['file_name'])
                         for r in inventory if r['id'] not in original], key=lambda r: r['image_id'])
    random.Random(20261031).shuffle(candidates)
    test = candidates[:a.test_images]
    plan = dict(search_images=validation[:16], confirmation_images=validation[16:],
                test=test, excluded_image_ids=sorted(original))
    if (run/'plan.json').exists() and json.loads((run/'plan.json').read_text()) != plan:
        raise ValueError('Input inventory changed; use a fresh --run-dir')
    write_json(run/'plan.json', plan)
    def prepare_fresh_images():
        from prepare import download
        from PIL import Image
        def materialize(row):
            dst = ROOT/'data/caption-fresh'/row['image']
            download('https://s3.amazonaws.com/images.cocodataset.org/val2014/'+Path(row['image']).name, dst)
            with Image.open(dst) as image:
                image.verify()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(materialize, test))
    if a.stage == 'prepare':
        prepare_fresh_images()
        print('Caption-domain inputs prepared. Copy the entire package for offline use.')
        return
    training = run/'caption-training'
    model = ROOT/'models/llava-1.5-7b-hf'
    if a.stage in ['all', 'train', 'extract']:
        if not (training/'protocol.json').exists():
            subprocess.run([sys.executable, '-u', str(ROOT/'caption_train.py'),
                '--package', str(ROOT), '--model', str(model), '--data-root', str(ROOT/'data'),
                '--annotations', str(annotations), '--out', str(training), '--images', '60'], check=True)
    if not (training/'direction.pt').exists():
        raise FileNotFoundError('Run train first')
    training_protocol = json.loads((training/'protocol.json').read_text())
    if training_protocol['source_sha256'] != sha(ROOT/'caption_train.py'):
        raise ValueError('Training source changed; use a fresh --run-dir')
    if a.stage in ['train', 'extract']:
        return
    direction_hash = sha(training/'direction.pt')
    features_hash = sha(training/'features.pt')
    selection_path = run/'selection.json'
    if a.stage in ['all', 'validate']:
        if not selection_path.exists():
            rows = infer(validation[:16], CONFIGS, ROOT/'data', model, training,
                         run/'search_predictions.jsonl', 128)
            metrics = score_rows(rows, ['base']+[c['name'] for c in CONFIGS], annotations)
            write_json(run/'search_metrics.json', metrics)
            valid = [c for c in CONFIGS if passes(metrics['base'], metrics[c['name']])]
            selected = min(valid, key=lambda c: (metrics[c['name']]['CHAIRs'],
                metrics[c['name']]['CHAIRi'], abs(c['alpha']))) if valid else None
            write_json(selection_path, dict(selected=selected, direction_sha256=direction_hash, features_sha256=features_hash, source_sha256=sha(Path(__file__)),
                plan_sha256=sha(run/'plan.json'), test_access=False))
        selection = json.loads(selection_path.read_text())
        if selection['selected'] is None:
            print('No candidate passed search validation; no final test was run.')
            return
        if selection['source_sha256'] != sha(Path(__file__)) or selection['direction_sha256'] != direction_hash or selection['features_sha256'] != features_hash or selection['plan_sha256'] != sha(run/'plan.json'):
            raise ValueError('Frozen training or plan changed')
        if not (run/'confirmation.json').exists():
            cfg = selection['selected']
            rows = infer(validation[16:], [cfg], ROOT/'data', model, training,
                         run/'confirmation_predictions.jsonl', 128)
            metrics = score_rows(rows, ['base', cfg['name']], annotations)
            write_json(run/'confirmation_metrics.json', metrics)
            write_json(run/'confirmation.json', dict(passed=passes(metrics['base'], metrics[cfg['name']]),
                selection_sha256=sha(selection_path), test_access=False))
        if a.stage == 'validate':
            return
    if a.stage in ['all', 'evaluate']:
        confirmation_path = run/'confirmation.json'
        if not confirmation_path.exists():
            raise FileNotFoundError('Run validate first')
        confirmation = json.loads(confirmation_path.read_text())
        if not confirmation['passed']:
            print('Candidate did not pass independent confirmation; no final test was run.')
            return
        selection = json.loads(selection_path.read_text())
        if confirmation['selection_sha256'] != sha(selection_path) or selection['source_sha256'] != sha(Path(__file__)) or selection['direction_sha256'] != direction_hash or selection['features_sha256'] != features_hash or selection['plan_sha256'] != sha(run/'plan.json'):
            raise ValueError('Frozen configuration changed')
        prepare_fresh_images()
        cfg = selection['selected']
        rows = infer([r['image'] for r in test], [cfg], ROOT/'data/caption-fresh', model, training,
                     run/'test_predictions_raw.jsonl', 128)
        renamed = []
        for row in rows:
            row = dict(row)
            row['aod_caption'] = row.pop(cfg['name']+'_caption')
            row['aod_tokens'] = row.pop(cfg['name']+'_tokens')
            renamed.append(row)
        (run/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in renamed))
        write_json(run/'metrics.json', score_rows(renamed, ['base', 'aod'], annotations))
        write_json(run/'execution.json', dict(completed=True, direction_sha256=direction_hash,
            selection_sha256=sha(selection_path), confirmation_sha256=sha(confirmation_path),
            plan_sha256=sha(run/'plan.json'), script_sha256=sha(Path(__file__)),
            test_images=len(test), prompt=PROMPT, max_new_tokens=128, finished=time.time()))
        print('Completed. Predictions and metrics:', run)


if __name__ == '__main__':
    main()
