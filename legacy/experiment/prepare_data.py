"""Materialize public snapshots; preserve original IDs and group-disjoint splits."""
import collections
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def assign_splits(rows, seed=42):
    groups = sorted({r['group'] for r in rows})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    mapping = {g: ('train' if i < int(.6*n) else 'val' if i < int(.8*n) else 'test')
               for i, g in enumerate(groups)}
    for r in rows:
        r['split'] = mapping[r['group']]
    return rows


def main():
    (DATA / 'images').mkdir(exist_ok=True)
    rows = []
    for path in sorted((DATA / 'hf').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for raw in batch.to_pylist():
                image = raw.pop('image')
                if path.name.startswith('pope'):
                    bench = 'pope_' + raw['category']
                    uid = bench + ':' + str(raw['question_id'])
                    group = 'pope:' + raw['image_source']
                    gt = int(raw['answer'].lower() == 'yes')
                    image_name = raw['image_source'] + '.jpg'
                else:
                    bench = 'hallusionbench'
                    uid = 'hb:' + ':'.join(str(raw[k]) for k in
                        ('category','subcategory','set_id','figure_id','question_id'))
                    group = 'hb:' + ':'.join(str(raw[k]) for k in ('category','subcategory','set_id'))
                    gt = int(raw['gt_answer'])
                    image_name = '_'.join(str(raw[k]) for k in
                        ('category','subcategory','set_id','figure_id')) + '.png'
                    assert (image is not None) == (str(raw['visual_input']) in {'1','2'})
                image_rel = None
                if image is not None:
                    image_rel = 'images/' + image_name
                    dst = DATA / image_rel
                    payload = image['bytes']
                    if dst.exists():
                        assert dst.read_bytes() == payload, f'Image collision: {dst}'
                    else:
                        dst.write_bytes(payload)
                rows.append(dict(id=uid, benchmark=bench, group=group, question=raw['question'],
                    gt=gt, image=image_rel, raw=raw))
    assert len(rows) == 10129, len(rows)
    assert len({r['id'] for r in rows}) == len(rows)
    hb = [r for r in rows if r['benchmark'] == 'hallusionbench']
    pope = [r for r in rows if r['benchmark'].startswith('pope')]
    rows = assign_splits(hb) + assign_splits(pope)
    # The official questions/labels must agree exactly with the mirror.
    official = json.loads((ROOT/'reference/HallusionBench/HallusionBench.json').read_text())
    lookup = {':'.join(str(r[k]) for k in ('category','subcategory','set_id','figure_id','question_id')):r
              for r in official}
    for r in hb:
        ref = lookup[r['id'][3:]]
        assert ref['question'] == r['question'] and int(ref['gt_answer']) == r['gt']
    for r in pope:
        name = r['benchmark'].removeprefix('pope_')
        p = ROOT/'reference/POPE/output/coco'/f'coco_pope_{name}.json'
        # Compared below in bulk; HF and original POPE can be distinct snapshots.
    path = DATA/'manifest.json'
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    report = {'n':len(rows), 'seed':42, 'split_unit':{'pope':'image_source across all splits',
        'hallusionbench':'category/subcategory/set_id (original and edited images together)'},
        'counts':dict(collections.Counter(r['benchmark']+'/'+r['split'] for r in rows)),
        'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'hf').glob('*.parquet')}}
    (DATA/'manifest_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
