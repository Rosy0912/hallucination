"""Recompute paired metrics without loading the model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import binary_metrics, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args(); rows=[json.loads(x) for x in a.predictions.read_text().splitlines() if x.strip()]
    result={k:binary_metrics(rows,col) for k,col in [('base','base_pred'),('aod','pred')]}
    result['by_type']={t:{k:binary_metrics([r for r in rows if r['typology']==t],col)
                         for k,col in [('base','base_pred'),('aod','pred')]}
                       for t in sorted({r['typology'] for r in rows})}
    if a.output:write_json(a.output,result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
