"""Build deterministic independent source ZIPs, excluding runtime data and weights."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def build():
    out=ROOT/'packages';out.mkdir(exist_ok=True);report={}
    for name in ['pope','hallusionbench','amber','chair']:
        target=out/(name+'.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
            for f in sorted((ROOT/name).rglob('*')):
                rel=f.relative_to(ROOT/name)
                if not f.is_file() or f.is_symlink() or rel.parts[0] in ['data','models','results','.venv']:continue
                if '__pycache__' in rel.parts or f.suffix=='.pyc' or f.name.startswith('._') or f.name=='.DS_Store':continue
                entry=zipfile.ZipInfo(str(f.relative_to(ROOT)),date_time=(2026,10,30,0,0,0))
                entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o100644<<16
                archive.writestr(entry,f.read_bytes())
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
            for f in ['run.sh','setup.sh','pipeline.py','prepare.py','requirements.txt','src/aod/data/amber.py']:
                assert name+'/'+f in archive.namelist()
        report[name]=dict(bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/'sha256.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':build()
