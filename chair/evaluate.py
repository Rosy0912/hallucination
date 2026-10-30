"""CHAIR scoring with original token rules, occurrence counts and both annotations."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tools.runtime import write_json


def score(evaluator, rows, caption_key):
    sentences=[]; mentions=hallucinations=hallucinated_caps=0
    for r in rows:
        words,nodes,idxs,raw=evaluator.caption_to_words(r[caption_key])
        gt=evaluator.imid_to_objects[r['image_id']]
        wrong=[(word,node) for word,node in zip(words,nodes) if node not in gt]
        mentions+=len(nodes);hallucinations+=len(wrong);hallucinated_caps+=bool(wrong)
        sentences.append(dict(image_id=r['image_id'],caption=r[caption_key],
                              generated_objects=nodes,ground_truth=sorted(gt),hallucinated_objects=wrong))
    return dict(n=len(rows),CHAIRs=100*hallucinated_caps/len(rows),
                CHAIRi=100*hallucinations/mentions if mentions else None,
                object_mentions=mentions,hallucinated_mentions=hallucinations,
                hallucinated_captions=hallucinated_caps,sentences=sentences)


def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--annotations',type=Path)
    source.add_argument('--ground-truth',type=Path,help='Previously exported official-rule object sets for these exact image IDs')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(x) for x in a.predictions.read_text().splitlines() if x.strip()]
    if not rows or len({r['image_id'] for r in rows})!=len(rows):
        raise ValueError('Empty predictions or duplicate image IDs')
    try:
        from .official_algorithm import CHAIR
    except ImportError:
        from official_algorithm import CHAIR
    evaluator=CHAIR([r['image_id'] for r in rows],str(a.annotations or ''))
    if a.annotations:
        known=set()
        for split in ['val','train']:
            known.update(r['id'] for r in json.loads((a.annotations/f'captions_{split}2014.json').read_text())['images'])
        if not {r['image_id'] for r in rows} <= known:
            raise ValueError('Image IDs missing from COCO annotation inventory')
        evaluator.get_annotations()
        write_json(a.output.with_name('ground_truth.json'),
                   {str(k):sorted(v) for k,v in evaluator.imid_to_objects.items()})
    else:
        ground_truth=json.loads(a.ground_truth.read_text())
        if {str(r['image_id']) for r in rows}!=set(ground_truth):
            raise ValueError('Prediction image IDs differ from exported ground truth')
        evaluator.imid_to_objects={int(k):set(v) for k,v in ground_truth.items()}
    result={mode:score(evaluator,rows,mode+'_caption') for mode in ['base','aod']}
    write_json(a.output,result)
    print(json.dumps({k:{m:v for m,v in vals.items() if m!='sentences'} for k,vals in result.items()},indent=2))

if __name__=='__main__':main()
