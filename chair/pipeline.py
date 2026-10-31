"""Self-contained public-data -> features -> training -> paired evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
BENCH=json.loads((ROOT/'benchmark.json').read_text())['benchmark']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['all','prepare','extract','train','validate','evaluate'],nargs='?',default='all')
    p.add_argument('--run-dir',default='results/default')
    p.add_argument('--aod-variant',choices=['caption','verified','question'],default='caption',help='Caption-domain AOD, optional object-verified composite, or question-domain direction')
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--coco-root',type=Path);p.add_argument('--hb-root',type=Path)
    p.add_argument('--amber-root',type=Path);p.add_argument('--annotations-root',type=Path)
    p.add_argument('--images',type=int,help='Final CHAIR images: caption default 200, question default 50')
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--test-images',type=int,default=200,help='Final image count: caption 1..200, verified 1..400; --images takes precedence')
    a=p.parse_args();os.chdir(ROOT)
    run=Path(a.run_dir)
    if run.is_absolute():raise ValueError('--run-dir must be relative to this package for portability')
    model='models/llava-1.5-7b-hf'
    def execute(script,*args):
        cmd=[sys.executable,'-u',script,*map(str,args)]
        print('RUN', ' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
    def prepare():
        args=[]
        for name in ['model_dir','coco_root','hb_root','amber_root','annotations_root']:
            value=getattr(a,name)
            if value:args.extend(['--'+name.replace('_','-'),value.resolve()])
        execute('prepare.py',*args)
    if a.stage in ['all','prepare']:prepare()
    if a.aod_variant=='verified':
        execute('verify_caption_pipeline.py',a.stage,'--run-dir',run,
                '--test-images',a.test_images if a.images is None else a.images,
                '--max-new-tokens',a.max_new_tokens)
        return
    if a.aod_variant=='caption':
        execute('caption_pipeline.py',a.stage,'--run-dir',run,
                '--test-images',a.test_images if a.images is None else a.images,
                '--max-new-tokens',a.max_new_tokens)
        return
    if a.stage=='validate':raise ValueError('Separate validate is available for --aod-variant caption')
    if a.stage=='prepare':return
    settings=dict(protocol_version=json.loads((ROOT/'benchmark.json').read_text())['protocol_version'],benchmark=BENCH,images=a.images,max_new_tokens=a.max_new_tokens,
                  test_images=a.test_images,answer_token_protocol='word-start-v3',model=model,training='fixed package protocol; validation selection only')
    run.mkdir(parents=True,exist_ok=True)
    config=run/'run_config.json'
    if a.stage!='evaluate' and config.exists() and json.loads(config.read_text())!=settings:raise ValueError('Run configuration changed; use a new --run-dir')
    if a.stage!='evaluate':config.write_text(json.dumps(settings,indent=2))
    features=run/'features'
    if BENCH in ['pope','hallusionbench','chair'] and a.stage in ['all','extract']:
        layers='13' if BENCH=='pope' else '24' if BENCH=='chair' else '1,4,8,12,16,20,24,28,31,32'
        benchmark='hallusionbench' if BENCH=='hallusionbench' else 'pope_mscoco'
        execute('experiment/extract_llava.py','--benchmark',benchmark,'--manifest','assets/manifest.json',
                '--layers',layers,'--out',features)
    if a.stage=='extract' and BENCH!='amber':return
    trained=run/'train';evaluation=run/'evaluation';experiment=run/'experiment'
    if a.stage in ['all','train','extract']:
        if BENCH=='pope':
            if not (trained/'results.json').exists():
                execute('experiment/user_gt_models.py','--source',features,'--out',trained,
                        '--layers','13','--variants','projective','--alphas','0','--seed','42')
        elif BENCH=='hallusionbench':
            if not (trained/'results.json').exists():
                execute('experiment/hb_aod_gt.py','--source',features,'--out',trained,
                        '--layers','1,4,8,12,16,20,24,28,31,32','--normalization','standard',
                        '--width','64','--balanced','--f1-guard')
        elif BENCH=='amber':
            images=200 if a.images is None else a.images
            if not 5<=images<=800:raise ValueError('AMBER --images is the extraction pool (5..800); use --test-images 0 for all remaining holdout images')
            if not (features/'extraction_protocol.json').exists():
                execute('run.py','--data-root','data/AMBER/data','--image-root','data/AMBER/image',
                        '--model',model,'--out',features,'--images',images,'--extract-only')
            if a.stage=='extract':return
            if not (experiment/'selection.json').exists():
                execute('optimize.py','--cache',features/'features.pt','--model',model,
                        '--data-root','data/AMBER/data','--image-root','data/AMBER/image',
                        '--out',experiment,'--test-images',a.test_images)
        elif BENCH=='chair':
            if not (experiment/'selection.json').exists():
                os.environ['NLTK_DATA']=str(ROOT/'data/nltk_data')
                execute('optimize.py','--cache',features/'features.pt','--data-root','data',
                        '--model',model,'--annotations','data/coco/annotations','--out',experiment,
                        '--test-images',50 if a.images is None else a.images,'--max-new-tokens',a.max_new_tokens)
    if BENCH in ['amber','chair'] and (experiment/'selection.json').exists() and not (experiment/'metrics.json').exists():
        selection=json.loads((experiment/'selection.json').read_text())
        if selection.get('selected',selection.get('config')) is not None:
            raise RuntimeError('Validation completed but final evaluation is incomplete; retain this run and use a fresh --run-dir')
    if a.stage in ['all','evaluate']:
        if BENCH=='pope' and not (evaluation/'metrics.json').exists():
            execute('infer.py','--source',features,'--trained',trained,'--out',evaluation)
        elif BENCH=='hallusionbench' and not (evaluation/'metrics.json').exists():
            execute('experiment/verify_hb_aod_gt.py','--source',features,'--experiment',trained,
                    '--model','llava','--out',evaluation)
        elif BENCH=='amber' and (experiment/'predictions.jsonl').exists():
            execute('evaluate.py',experiment/'predictions.jsonl','--output',experiment/'metrics.json')
        elif BENCH=='chair' and (experiment/'predictions.jsonl').exists():
            os.environ['NLTK_DATA']=str(ROOT/'data/nltk_data')
            execute('evaluate.py',experiment/'predictions.jsonl','--annotations','data/coco/annotations',
                    '--output',experiment/'metrics.official.json')
    if BENCH in ['amber','chair'] and not (experiment/'predictions.jsonl').exists():
        print('No candidate passed validation; inspect selection.json. No test claim is made.',flush=True)
    print('DONE. Results:',run.resolve(),flush=True)

if __name__=='__main__':main()
