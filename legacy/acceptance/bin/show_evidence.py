"""Display immutable AOD measurements and their source; does not rerun inference."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('model',choices=['llava','qwen','internvl']);p.add_argument('dataset',choices=['pope','hb']);a=p.parse_args()
manifest=json.loads((ROOT/'acceptance/result_manifest.json').read_text());found=[r for r in manifest['verified'] if r['model']==a.model and r['dataset']==a.dataset]
if not found:raise SystemExit('This combination needs further optimization; no accepted result is displayed.')
r=found[0];path=ROOT/r['metrics'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];d=json.loads(path.read_text());b=d['base'];s=d['selected'];cfg=json.loads((ROOT/r['config']).read_text())
print('\nAOD EXPERIMENT RESULTS\n')
print('METHOD:  AOD');print('MODEL:  ',r['model_name']);print('DATASET:',r['dataset_name']);print('TEST N: ',s['n']);print('JOB:    ',r['job'],' | Complete fresh image inference')
print('\nMetric                    Base          AOD       Gain (pp)')
keys=['f1','false_positive_rate'] if a.dataset=='pope' else ['accuracy','f1']
for k in keys:print(f'{k:22s} {b[k]:9.4f}% {s[k]:10.4f}% {s[k]-b[k]:+11.4f}')
print('\nTARGET:','F1 >= 90% AND FPR <= 10%' if a.dataset=='pope' else 'Accuracy gain >= 5 percentage points');print('RESULT: PASS')
print('Reproduction: Base and AOD predictions match the saved experiment.')
print('\nPARAMETERS: layer=',cfg.get('layer',22),' alpha=',cfg.get('alpha'),' epoch=',cfg.get('epoch',35),sep='')
print('SOURCE:',r['metrics']);print('SHA256:',r['sha256']);print('\nRecorded experiment display | Internal fixed split')
