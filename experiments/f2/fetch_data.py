"""Fetch MNLI (GLUE) and SNLI parquet files at pinned dataset revisions, sha256-verified.

    ./venv/bin/python experiments/f2/fetch_data.py      -> experiments/f2/raw/*.parquet (gitignored)

MNLI validation (matched + mismatched) feeds val_ood (step 0.3); MNLI train and SNLI train feed the
natural low-overlap augmentation (step 3.2). Neither corpus is in any earlier split except SNLI
validation (used by clean DEV and CANNOT phase 2), which is not fetched here.
"""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "f1"))
from fetch_models import _get  # noqa: E402  (ranged GET with retries)

FILES = {   # local name: (dataset, revision, path)
    "mnli_validation_matched.parquet": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "mnli/validation_matched-00000-of-00001.parquet"),
    "mnli_validation_mismatched.parquet": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "mnli/validation_mismatched-00000-of-00001.parquet"),
    "mnli_train.parquet": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "mnli/train-00000-of-00001.parquet"),
    "snli_train.parquet": ("stanfordnlp/snli", "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b", "plain_text/train-00000-of-00001.parquet"),
}
CHUNK = 2 * 2 ** 20


def main() -> None:
    out = HERE / "raw"
    out.mkdir(exist_ok=True)
    manifest = {}
    for name, (ds, rev, path) in FILES.items():
        tree = {f["path"]: f for f in json.loads(_get(f"https://huggingface.co/api/datasets/{ds}/tree/{rev}/{path.rsplit('/', 1)[0]}"))}
        f = tree[path]
        sha = f["lfs"]["oid"]
        dest = out / name
        if not (dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha):
            url = f"https://huggingface.co/datasets/{ds}/resolve/{rev}/{path}"
            ranges = [(s, min(s + CHUNK, f["size"]) - 1) for s in range(0, f["size"], CHUNK)]
            with ThreadPoolExecutor(32) as ex:
                data = b"".join(ex.map(lambda r: _get(url, r), ranges))
            assert hashlib.sha256(data).hexdigest() == sha, name
            dest.write_bytes(data)
        manifest[name] = {"dataset": ds, "revision": rev, "path": path, "sha256": sha, "size": f["size"]}
        print(name, "ok", flush=True)
    (HERE / "results" / "raw_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
