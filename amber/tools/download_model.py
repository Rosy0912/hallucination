"""Optional public mirror download, verified against pinned Hugging Face hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

def matches(path, row):
    if not path.is_file() or path.stat().st_size != row['size']:
        return False
    digest = hashlib.sha256() if 'lfs' in row else hashlib.sha1(b'blob ' + str(row['size']).encode() + b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    expected = row['lfs']['sha256'] if 'lfs' in row else row['blobId']
    return digest.hexdigest() == expected

def download_verified_mirror(destination, manifest_path):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    def fetch(row):
        name = row['rfilename']
        target = destination / name
        if matches(target, row):
            return
        url = 'https://modelscope.cn/models/' + manifest['repo'] + '/resolve/master/' + name
        temporary = target.with_suffix(target.suffix + '.mirror.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
                    for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                        stream.write(chunk)
                if not matches(temporary, row):
                    raise ValueError('Mirror differs from pinned model revision: ' + name)
                temporary.replace(target)
                print('Verified model file:', name, flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, manifest['files']))
    (destination / 'mirror_verification.json').write_text(json.dumps(dict(
        repo=manifest['repo'], revision=manifest['revision'],
        source='ModelScope mirror, all bytes verified against pinned Hugging Face git/LFS hashes'), indent=2))
