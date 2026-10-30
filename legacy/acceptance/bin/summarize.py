"""Print fresh-run measurements. Never substitutes historical scores."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
path=a.run/'metrics.json'
if not path.exists():path=a.run/'evaluation/metrics.json'
if not path.exists():raise SystemExit('No completed metrics yet. Check squeue and the job log.')
d=json.loads(path.read_text());print('RESULT',path)
for k in ['n','accuracy','f1','false_positive_rate','false_discovery_rate','tp','tn','fp','fn']:print(f'{k}: Base={d["base"][k]} Selected={d["selected"][k]}')
d.setdefault('accuracy_gain_pp',d['selected']['accuracy']-d['base']['accuracy'])
d.setdefault('f1_gain_pp',d['selected']['f1']-d['base']['f1'])
if 'criterion_met' not in d and 'protocol' in d:
 dataset=d['protocol'].get('dataset','')
 if not dataset:dataset=str(d['protocol'].get('source_split','')).lower()
 if 'pope' in dataset:d['criterion_met']=d['selected']['f1']>=90 and d['selected']['false_positive_rate']<=10
 elif 'hb' in dataset or 'hallusion' in dataset:d['criterion_met']=d['accuracy_gain_pp']>=5
for k in ['accuracy_gain_pp','f1_gain_pp','complete_test','criterion_met','base_predictions_exact','selected_predictions_exact','features_exact','retrained','elapsed_seconds']:
 if k in d:print(k,d[k])
