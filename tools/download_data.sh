#!/usr/bin/env bash
# Run on a machine with Google Drive / raw.githubusercontent.com access.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
mkdir -p data/AMBER/data/query data/coco
curl -fL --retry 3 https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/annotations.json -o data/AMBER/data/annotations.json
curl -fL --retry 3 https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/query/query_all.json -o data/AMBER/data/query/query_all.json
"$PYTHON" -m gdown 1MaCHgtupcZUjf007anNl4_MV0o4DjXvl -O data/AMBER/images.zip
# The canonical COCO endpoint uses HTTP (its HTTPS certificate mismatches).
curl -fL --retry 3 http://images.cocodataset.org/annotations/annotations_trainval2014.zip -o data/coco/annotations.zip
"$PYTHON" - <<'PY'
import zipfile
for src,dst in [('data/AMBER/images.zip','data/AMBER'),('data/coco/annotations.zip','data/coco')]:
    with zipfile.ZipFile(src) as z:
        assert z.testzip() is None
        z.extractall(dst)
PY
"$PYTHON" -m nltk.downloader -d data/nltk_data punkt_tab
