"""Offline metric recomputation for the two original frozen benchmark runs."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import binary_metrics,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();rows=[json.loads(x) for x in a.predictions.read_text().splitlines() if x.strip()]
    result={k:binary_metrics(rows,c) for k,c in [('base','base_pred'),('aod','pred')]}
    if all(r.get('benchmark')=='hallusionbench' for r in rows):
        for mode,key in [('base','base_pred'),('aod','pred')]:
            figures,pairs=defaultdict(list),defaultdict(list)
            for row in rows:
                raw=row['raw'];correct=row[key]==row['gt']
                prefix=tuple(str(raw[k]) for k in ['category','subcategory','set_id'])
                if not(raw['category']=='VS' and str(raw['figure_id'])=='0'):
                    figures[prefix+(str(raw['figure_id']),)].append(correct)
                pairs[prefix+(str(raw['question_id']),)].append(correct)
            result[mode].update(figure_accuracy=100*sum(all(v) for v in figures.values())/max(1,len(figures)),
                                pair_accuracy=100*sum(all(v) for v in pairs.values())/max(1,len(pairs)),
                                n_figures=len(figures),n_pairs=len(pairs))
    if a.output:write_json(a.output,result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
