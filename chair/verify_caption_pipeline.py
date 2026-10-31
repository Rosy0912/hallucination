"""Optional AOD + image/object verifier + selective caption rewrite.

Train both components from public data; select on development images; confirm on
fresh images; evaluate one frozen candidate. All measurements stay in --run-dir.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.runtime import write_json, sha
from caption_pipeline import infer, score_rows

ANCHOR = dict(name='anchor', mode='projection_cd', alpha=2., beta=.5)
THRESHOLDS = [-1., 0., .5]
NAMES = ['base', 'anchor', 'verified_base', 'aod']


def passes(metrics):
    candidate = metrics['aod']
    return (all(candidate[key] is not None and metrics[mode][key] is not None
                and candidate[key] < metrics[mode][key]
                for mode in ['base', 'anchor'] for key in ['CHAIRs', 'CHAIRi'])
            and all(candidate[key] >= .9 * metrics['base'][key]
                    for key in ['mean_tokens', 'correct_unique_objects_per_image', 'object_recall']))


def generate(rows, thresholds, data_root, model_path, training, verifier, output):
    protocol = dict(images=[row['image'] for row in rows], thresholds=thresholds,
                    captions_sha256=sha(output.parent/'initial_predictions.jsonl'),
                    verifier_sha256=sha(verifier), direction_sha256=sha(training/'direction.pt'),
                    decoder=ANCHOR, max_new_tokens=128,
                    source_sha256=sha(Path(__file__)), helper_sha256=sha(ROOT/'object_verifier.py'))
    protocol_path = output.with_suffix('.protocol.json')
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Inference inputs changed; choose a fresh --run-dir')
    if output.exists():
        if not protocol_path.exists():
            raise ValueError('Missing inference protocol')
        saved = [json.loads(line) for line in output.read_text().splitlines() if line]
        if [row['image'] for row in saved] != protocol['images']:
            raise RuntimeError('Incomplete generation; preserve this run and choose a fresh --run-dir')
        return saved
    import torch
    from tools.runtime import load_llava
    from aod.vlm.intervention import AODDecodeConfig
    from object_verifier import ObjectVerifier
    direction = torch.load(training/'direction.pt', weights_only=False, map_location='cpu')['direction']
    config = AODDecodeConfig(layer=24, direction=direction, alpha=2., beta=.5, apc_alpha=.1, mode='cd')
    model, processor = load_llava(model_path)
    checker = ObjectVerifier(verifier, model, processor)
    write_json(protocol_path, protocol)
    result = []
    with torch.inference_mode(), output.open('x') as handle:
        for index, row in enumerate(rows):
            image = data_root/row['image']
            checks = checker.checks(image, row['anchor_caption'])
            record = dict(row, variants={})
            for threshold in thresholds:
                record['variants'][str(threshold)] = checker.rewrite(image, row, checks, threshold, config)
            result.append(record)
            handle.write(json.dumps(record)+'\n')
            handle.flush()
            print('Object verification', index+1, len(rows), flush=True)
    del checker, model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'prepare', 'extract', 'train', 'validate', 'evaluate'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--test-images', type=int, default=400)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.test_images <= 400 or args.max_new_tokens != 128:
        raise ValueError('Use 1..400 test images and exactly 128 tokens')
    os.environ['NLTK_DATA'] = str(ROOT/'data/nltk_data')
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    annotations = ROOT/'data/coco/annotations'
    manifest = json.loads((ROOT/'assets/manifest.json').read_text())
    original = {int(Path(row['image']).stem.rsplit('_', 1)[-1]) for row in manifest}
    validation = sorted({row['image'] for row in manifest if row['split'] == 'val'})
    random.Random(20261030).shuffle(validation)
    inventory = json.loads((annotations/'captions_val2014.json').read_text())['images']
    candidates = sorted([dict(image_id=row['id'], image='images/'+row['file_name'])
                         for row in inventory if row['id'] not in original], key=lambda row: row['image_id'])
    random.Random(20261031).shuffle(candidates)
    excluded = original | {row['image_id'] for row in candidates[:200]}
    candidates = sorted([row for row in candidates if row['image_id'] not in excluded], key=lambda row: row['image_id'])
    random.Random(20261101).shuffle(candidates)
    plan = dict(search_images=validation[:16], confirmation=candidates[:80],
                test=candidates[80:80+args.test_images], excluded_image_ids=sorted(excluded),
                thresholds=THRESHOLDS, anchor=ANCHOR, max_new_tokens=128)
    if (run/'plan.json').exists() and json.loads((run/'plan.json').read_text()) != plan:
        raise ValueError('Plan changed; choose a fresh --run-dir')
    write_json(run/'plan.json', plan)
    fresh = ROOT/'data/verified-caption-fresh'
    def materialize():
        from prepare import download
        from PIL import Image
        def one(row):
            path = fresh/row['image']
            download('https://s3.amazonaws.com/images.cocodataset.org/val2014/'+Path(row['image']).name, path)
            with Image.open(path) as image:
                image.verify()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one, plan['confirmation']+plan['test']))
    if args.stage == 'prepare':
        materialize()
        return
    def execute(script, *arguments):
        subprocess.run([sys.executable, '-u', str(ROOT/script), *map(str, arguments)], check=True, cwd=ROOT)
    model = ROOT/'models/llava-1.5-7b-hf'
    training = run/'caption-training'
    features = run/'verifier-features'
    trained = run/'verifier-training'
    verifier = trained/'pope_mscoco_L13_projective_gt_s42.pt'
    if args.stage in ['all', 'extract', 'train']:
        if not (training/'protocol.json').exists():
            execute('caption_train.py', '--package', ROOT, '--model', model, '--data-root', ROOT/'data',
                    '--annotations', annotations, '--out', training, '--images', '60')
        if not (features/'extraction_completed.json').exists():
            execute('experiment/extract_llava.py', '--benchmark', 'pope_mscoco', '--manifest', ROOT/'assets/manifest.json',
                    '--layers', '13', '--out', features)
        if not (trained/'results.json').exists():
            execute('experiment/user_gt_models.py', '--source', features, '--out', trained,
                    '--layers', '13', '--variants', 'projective', '--alphas', '0', '--seed', '42')
    if not verifier.exists() or not (training/'direction.pt').exists():
        raise FileNotFoundError('Run train first')
    if json.loads((training/'protocol.json').read_text())['source_sha256'] != sha(ROOT/'caption_train.py'):
        raise ValueError('Caption training source changed; choose a fresh --run-dir')
    if args.stage in ['extract', 'train']:
        return
    frozen = dict(plan_sha256=sha(run/'plan.json'), direction_sha256=sha(training/'direction.pt'),
                  verifier_sha256=sha(verifier), source_sha256=sha(Path(__file__)),
                  helper_sha256=sha(ROOT/'object_verifier.py'), decoder_sha256=sha(ROOT/'caption_pipeline.py'))
    def phase(name, samples, data_root, thresholds):
        output = run/name
        output.mkdir(exist_ok=True)
        paths = [row['image'] for row in samples]
        initial = infer(paths, [ANCHOR], data_root, model, training, output/'initial_predictions.jsonl', 128)
        rows = generate(initial, thresholds, data_root, model, training, verifier, output/'predictions.jsonl')
        metrics = {}
        for threshold in thresholds:
            flat = [dict(row, **row['variants'][str(threshold)]) for row in rows]
            metrics[str(threshold)] = score_rows(flat, NAMES, annotations)
        write_json(output/'metrics.json', metrics)
        return metrics
    selection_path = run/'selection.json'
    if args.stage in ['all', 'validate']:
        if not selection_path.exists():
            metrics = phase('development', [dict(image=image) for image in plan['search_images']], ROOT/'data', THRESHOLDS)
            eligible = [threshold for threshold in THRESHOLDS if passes(metrics[str(threshold)])]
            selected = min(eligible, key=lambda threshold: (metrics[str(threshold)]['aod']['CHAIRs'],
                           metrics[str(threshold)]['aod']['CHAIRi'], threshold)) if eligible else None
            write_json(selection_path, dict(selected=selected, frozen=frozen, test_access=False))
        selection = json.loads(selection_path.read_text())
        if selection['frozen'] != frozen:
            raise ValueError('Frozen training or source changed')
        if selection['selected'] is None:
            print('No candidate passed development; final test not run.')
            return
        if not (run/'confirmation.json').exists():
            materialize()
            metrics = phase('confirmation', plan['confirmation'], fresh, [selection['selected']])
            write_json(run/'confirmation.json', dict(passed=passes(metrics[str(selection['selected'])]),
                       selection_sha256=sha(selection_path), test_access=False))
        if args.stage == 'validate':
            return
    if args.stage in ['all', 'evaluate']:
        selection = json.loads(selection_path.read_text())
        confirmation = json.loads((run/'confirmation.json').read_text())
        if selection['frozen'] != frozen or confirmation['selection_sha256'] != sha(selection_path):
            raise ValueError('Frozen configuration changed')
        if not confirmation['passed']:
            print('Confirmation did not pass; final test not run.')
            return
        materialize()
        phase('test', plan['test'], fresh, [selection['selected']])
        write_json(run/'execution.json', dict(completed=True, frozen=frozen, test_images=args.test_images,
                   method='AOD + POPE-supervised object verifier + selective rewrite',
                   selection_sha256=sha(selection_path), confirmation_sha256=sha(run/'confirmation.json')))
        print('Completed. Private local outputs:', run)


if __name__ == '__main__':
    main()
