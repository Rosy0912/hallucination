"""CHAIR scoring with original token rules, occurrence counts and both annotations."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
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
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(x) for x in a.predictions.read_text().splitlines() if x.strip()]
    if not rows or len({r['image_id'] for r in rows})!=len(rows):
        raise ValueError('Empty predictions or duplicate image IDs')
    known=set()
    for split in ['val','train']:
        known.update(r['id'] for r in json.loads((a.annotations/f'captions_{split}2014.json').read_text())['images'])
    if not {r['image_id'] for r in rows} <= known:
        raise ValueError('Image IDs missing from COCO annotation inventory')
    from chair.official_algorithm import CHAIR
    evaluator=CHAIR([r['image_id'] for r in rows],str(a.annotations));evaluator.get_annotations()
    result={mode:score(evaluator,rows,mode+'_caption') for mode in ['base','aod']}
    write_json(a.output,result)
    print(json.dumps({k:{m:v for m,v in vals.items() if m!='sentences'} for k,vals in result.items()},indent=2))

if __name__=='__main__':main()
