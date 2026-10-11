"""Download a pinned model revision with parallel HTTP range requests, verified by sha256.

The Hugging Face CDN throttles each connection to ~65 KB/s on this machine (measured), so a
1.6 GB model takes ~7 h over one connection. This splits each large file across many ranged
requests, then checks the result against the hub's LFS sha256 for that exact revision.
Files land in f1/cache/models/<key>/ and nli_models.Scorer loads from there.

    ./venv/bin/python experiments/f1/fetch_models.py KEY [KEY ...]
"""

from __future__ import annotations

import hashlib
import shutil
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from core import CACHE

WANT = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
        "merges.txt", "spm.model", "vocab.txt", "added_tokens.json", "model.safetensors")
CHUNK = 8 * 2 ** 20
WORKERS = 48


def _get(url: str, rng: tuple[int, int] | None = None, tries: int = 8) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if rng is None or len(data) == rng[1] - rng[0] + 1:
                return data
        except Exception:
            if attempt == tries - 1:
                raise
    raise IOError(f"short read {url} {rng}")


def fetch(key: str) -> None:
    from nli_models import MODELS
    name, rev, _ = MODELS[key]
    out = CACHE / "models" / key
    out.mkdir(parents=True, exist_ok=True)
    tree = json.loads(_get(f"https://huggingface.co/api/models/{name}/tree/{rev}"))
    files = {f["path"]: f for f in tree if f["type"] == "file"}
    if "model.safetensors" not in files:          # older repos ship only pytorch_model.bin
        files["model.safetensors"] = files.pop("pytorch_model.bin")
        files["model.safetensors"]["_src"] = "pytorch_model.bin"
    manifest = {"hub_id": name, "revision": rev, "files": {}}
    for path in WANT:
        if path not in files:
            continue
        f = files[path]
        src = f.get("_src", path)
        dest = out / src
        sha = (f.get("lfs") or {}).get("oid")
        if dest.exists() and dest.stat().st_size == f["size"]:
            manifest["files"][src] = {"size": f["size"], "sha256": sha}
            continue
        url = f"https://huggingface.co/{name}/resolve/{rev}/{src}"
        if f["size"] < CHUNK:
            data = _get(url)
        else:   # each chunk persisted as it lands, so a killed run resumes where it stopped
            parts = out / f".{src}.parts"
            parts.mkdir(exist_ok=True)
            ranges = [(s, min(s + CHUNK, f["size"]) - 1) for s in range(0, f["size"], CHUNK)]

            def one(r):
                part = parts / f"{r[0]:012d}"
                if not (part.exists() and part.stat().st_size == r[1] - r[0] + 1):
                    part.write_bytes(_get(url, r))

            with ThreadPoolExecutor(WORKERS) as ex:
                list(ex.map(one, ranges))
            data = b"".join((parts / f"{r[0]:012d}").read_bytes() for r in ranges)
        assert len(data) == f["size"], (src, len(data), f["size"])
        if sha:
            got = hashlib.sha256(data).hexdigest()
            assert got == sha, f"{src}: sha256 {got} != hub {sha}"
        dest.write_bytes(data)
        if (out / f".{src}.parts").exists():
            shutil.rmtree(out / f".{src}.parts")
        manifest["files"][src] = {"size": f["size"], "sha256": sha}
        print(key, src, f["size"], "ok", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    for k in sys.argv[1:]:
        fetch(k)
