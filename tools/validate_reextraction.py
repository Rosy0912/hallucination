"""Check semantic identity; this does NOT relax frozen-checkpoint SHA checks."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import binary_metrics,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('benchmark',choices=['pope','hallusionbench'])
    p.add_argument('features',type=Path);a=p.parse_args()
    import torch
    root=Path(__file__).resolve().parents[1]
    cache=torch.load(a.features,map_location='cpu',weights_only=False)
    expected=json.loads((root/'legacy/resources'/f'{a.benchmark}_manifest.json').read_text())
    rows=cache['records'];assert len(rows)==len(expected)
    for r,e in zip(rows,expected):
        assert all(r[k]==e[k] for k in ['id','gt','image','question','group','split'])
    print(json.dumps(dict(semantic_manifest_match=True,features_sha256=sha(a.features),
                          test_base=binary_metrics([r for r in rows if r['split']=='test']),
                          frozen_checkpoint_compatibility='not certified; original feature SHA is still required'),indent=2))
if __name__=='__main__':main()
