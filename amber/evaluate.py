"""Recompute paired metrics without loading the model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tools.runtime import binary_metrics, write_json

def evaluate_records(rows):
    # AMBER's official discriminative evaluator treats "No" as positive.
    # POPE uses "Yes" as positive. Keep both definitions explicitly named.
    def score(subset,key,epsilon=0.0001):
        inverted=[dict(id=r['id'],gt=1-r['gt'],pred=1-r[key]) for r in subset]
        m=binary_metrics(inverted)
        precision,recall=round(m['precision'],1),round(m['recall'],1)
        m['positive_label']='no'
        m['official_rounded']=dict(accuracy=round(m['accuracy'],1),precision=precision,recall=recall,
            f1=round(2*(precision/100)*(recall/100)/((precision+recall)/100+epsilon)*100,1))
        return m
    result={k:score(rows,col) for k,col in [('base','base_pred'),('aod','pred')]}
    result['by_type']={t:{k:score([r for r in rows if r['typology']==t],col,0.001 if t=='existence' else 0.0001)
                         for k,col in [('base','base_pred'),('aod','pred')]}
                      for t in sorted({r['typology'] for r in rows})}
    result['yes_positive_auxiliary']={k:binary_metrics(rows,col) for k,col in [('base','base_pred'),('aod','pred')]}
    result['metric_definition']='No-positive confusion matrix and exact F1; official_rounded also reproduces upstream rounding and epsilon.'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args(); rows=[json.loads(x) for x in a.predictions.read_text().splitlines() if x.strip()]
    result=evaluate_records(rows)
    if a.output:write_json(a.output,result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
