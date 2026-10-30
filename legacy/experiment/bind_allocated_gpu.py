"""Bind only Slurm-allocated GPU device minors, never raw NVML indices.

Run this wrapper INSIDE srun. It changes only its own process and descendants.
Unknown GRES forms/missing or duplicate devices fail closed. No config is edited.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def parse_indices(value):
    if not value:
        raise ValueError('No Slurm GPU allocation is present')
    result=[]
    for part in value.split(','):
        if re.fullmatch(r'\d+',part):result.append(int(part))
        elif re.fullmatch(r'\d+-\d+',part):
            lo,hi=map(int,part.split('-'))
            if not 0<=lo<=hi<1024:raise ValueError('Invalid GPU index range')
            result.extend(range(lo,hi+1))
        else:raise ValueError('Unsupported SLURM GPU index format')
    if len(set(result))!=len(result):raise ValueError('Duplicate Slurm GPU index')
    return result


def resolve_allocated(node,indices,gres_text,visible_devices):
    records=[]
    for raw in gres_text.splitlines():
        line=raw.split('#',1)[0].strip()
        if not line:continue
        fields=dict(part.split('=',1) for part in line.split() if '=' in part)
        if fields.get('NodeName')!=node or fields.get('Name')!='gpu':continue
        # Exact-node, one-device lines are the verified Lumia configuration.
        file=fields.get('File','')
        match=re.fullmatch(r'/dev/nvidia(\d+)',file)
        if not match:raise ValueError('Unsupported GRES File expression; require explicit device paths')
        records.append({'gres_line':line,'device':file,'minor':int(match.group(1)),'type':fields.get('Type')})
    if not records:raise ValueError('No explicit GPU GRES entries for this node')
    by_minor={}
    for row in visible_devices:
        if row['minor'] in by_minor:raise ValueError('Duplicate NVML device minor')
        by_minor[row['minor']]=row
    assigned=[]
    for index in indices:
        if not 0<=index<len(records):raise ValueError('Slurm GRES index is outside configured devices')
        item=records[index]
        if item['minor'] not in by_minor:raise ValueError('Allocated GPU is not visible; do not substitute another GPU')
        info=by_minor[item['minor']]
        if not info['uuid'].startswith('GPU-'):raise ValueError('Invalid allocated GPU UUID')
        assigned.append(dict(item,slurm_index=index,**info))
    if len({x['uuid'] for x in assigned})!=len(assigned):raise ValueError('Duplicate allocated UUID')
    return assigned


def mib(value):
    m=re.fullmatch(r'([0-9.]+) MiB',value or '')
    if not m:raise ValueError('Unrecognized NVML memory quantity')
    return float(m.group(1))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--expected-count',type=int,required=True)
    parser.add_argument('--min-free-mib',type=float,required=True)
    parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Must run inside a Slurm allocation')
    node=os.environ.get('SLURMD_NODENAME')
    if not node:raise ValueError('Missing SLURMD_NODENAME; do not guess host mapping')
    raw_ids=os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
    indices=parse_indices(raw_ids)
    if len(indices)!=args.expected_count:raise ValueError('Actual allocated GPU count differs from requested count')
    cp=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,timeout=15,check=True)
    xml=ET.fromstring(cp.stdout)
    visible=[{'uuid':g.findtext('uuid'),'name':g.findtext('product_name'),
              'minor':int(g.findtext('minor_number')),'pci_bus_id':g.findtext('pci/pci_bus_id'),
              'free_mib':mib(g.findtext('fb_memory_usage/free')),
              'used_mib':mib(g.findtext('fb_memory_usage/used'))} for g in xml.findall('gpu')]
    gres=Path('/etc/slurm/gres.conf').read_text()
    assigned=resolve_allocated(node,indices,gres,visible)
    if any(row['free_mib']<args.min_free_mib for row in assigned):
        raise RuntimeError('An allocated GPU lacks the required free memory')
    old_mask=os.environ.get('CUDA_VISIBLE_DEVICES')
    mask=','.join(row['uuid'] for row in assigned)
    os.environ['CUDA_VISIBLE_DEVICES']=mask
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    # Import only AFTER binding, so CUDA never initializes with the stale mask.
    import torch
    if torch.cuda.device_count()!=args.expected_count:raise RuntimeError('Bound GPU count mismatch')
    observed=[]
    for index,item in enumerate(assigned):
        prop=torch.cuda.get_device_properties(index)
        uuid='GPU-'+str(prop.uuid).removeprefix('GPU-')
        if uuid!=item['uuid']:raise RuntimeError('Torch UUID differs from allocated device UUID')
        free,total=torch.cuda.mem_get_info(index)
        if free/2**20<args.min_free_mib:raise RuntimeError('Post-CUDA-init free memory below required floor')
        observed.append({'cuda_index':index,'uuid':uuid,'name':prop.name,
                         'free_mib':free/2**20,'total_mib':total/2**20})
    audit={'time_unix':time.time(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ.get('SLURM_STEP_ID'),
           'node':node,'allocation_indices':indices,'original_cuda_visible_devices':old_mask,
           'bound_cuda_visible_devices':mask,'assigned_gpus':assigned,'torch_gpus':observed,
           'excluded_visible_uuids':[r['uuid'] for r in visible if r['uuid'] not in {a['uuid'] for a in assigned}],
           'minimum_free_mib':args.min_free_mib,'status':'verified','probe_only':args.probe_only}
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    if args.audit.exists():
        args.audit = args.audit.with_name(args.audit.stem + '.restart-' + str(time.time_ns()) + args.audit.suffix)
    with args.audit.open('x') as f:f.write(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'gpu_binding':'verified','job_id':audit['job_id'],'node':node,
                      'uuids':[r['uuid'] for r in assigned],'audit':str(args.audit)}),flush=True)
    if args.probe_only:return
    command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:raise ValueError('A target command is required unless probe-only')
    os.execvpe(command[0],command,os.environ)


if __name__=='__main__':main()
