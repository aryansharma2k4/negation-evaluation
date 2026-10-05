"""Step 4: three independent similarity-rating passes by a local LLM.

    ./venv/bin/python experiments/pattern/label.py            # all three passes
    ./venv/bin/python experiments/pattern/label.py --pass 2   # one pass

Reads ONLY data/to_label.jsonl (id, a, b). Never metadata.jsonl: the labeller
must not learn a pair's family, cue count, or that the project is about
negation -- otherwise any pattern found later is the one it was handed.

Pass 1  (A, B) in to_label order
Pass 2  (B, A) same order         -- isolates the position effect
Pass 3  (A, B) reshuffled order   -- different batch neighbours

Items go to the model five at a time. That is what makes pass 3 a real
perturbation: at temperature 0 a one-item prompt is deterministic, so a
reshuffle would reproduce pass 1 exactly and inflate agreement to 1.0. With
batches, the reshuffle changes each item's neighbours, and pass 1 vs pass 3
measures how much the context moves the score.

Each pass writes data/labels_pass{k}.jsonl and reads nothing but to_label.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

MODEL = "qwen2.5:7b"
OLLAMA = "http://localhost:11434/api/generate"
BATCH = 5
#: pass -> (swap A/B, shuffle seed or None for to_label order, sampling seed)
PASSES = {1: (False, None, 101), 2: (True, None, 202), 3: (False, 303, 303)}

PROMPT_HEAD = """You rate how similar in meaning two sentences are, on a 0-5 scale.

5 = completely equivalent
4 = mostly equivalent, minor unimportant detail differs
3 = roughly equivalent, some important information differs
2 = not equivalent but share the same topic
1 = not equivalent, topics barely related
0 = completely dissimilar

Worked examples:
Sentence 1: The committee approved the budget on Friday.
Sentence 2: On Friday, the budget was approved by the committee.
Score: 5 (same meaning, different wording)

Sentence 1: Maria flew to Paris for a conference last week.
Sentence 2: Maria flew to Rome for a conference last week.
Score: 3 (same kind of event, but the destination is important and differs)

Sentence 1: The river floods every spring after heavy rain.
Sentence 2: Farmers usually plant rice in the spring.
Score: 1 (both mention spring, otherwise unrelated)

Rate each pair below on its own; the pairs are unrelated to each other.
Answer with JSON only, in this form: {"scores": [{"pair": 1, "score": 4}, ...]}
One integer score from 0 to 5 per pair, for every pair.

"""


def load_items() -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "to_label.jsonl").read_text().splitlines() if l.strip()]
    for row in rows:
        assert set(row) == {"id", "a", "b"}, f"labelling input carries extra fields: {set(row)}"
    return rows


def build_prompt(pairs: list[tuple[str, str]]) -> str:
    body = "".join(f"Pair {i}\nSentence 1: {a}\nSentence 2: {b}\n\n"
                   for i, (a, b) in enumerate(pairs, 1))
    return PROMPT_HEAD + body


def ollama(prompt: str, seed: int) -> str:
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0, "seed": seed, "num_predict": 200, "num_ctx": 4096},
    }).encode()
    request = urllib.request.Request(OLLAMA, payload, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["response"]


def parse_scores(text: str, n: int) -> list[int | None]:
    try:
        entries = json.loads(text)["scores"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [None] * n
    out: list[int | None] = [None] * n
    for entry in entries if isinstance(entries, list) else []:
        try:
            k, score = int(entry["pair"]), float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= k <= n and 0 <= score <= 5 and out[k - 1] is None:
            out[k - 1] = int(round(score))
    return out


def pass_order(items: list[dict], pass_no: int) -> list[dict]:
    _, shuffle_seed, _ = PASSES[pass_no]
    order = list(items)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    return order


def run_pass(items: list[dict], pass_no: int, ask: Callable[[str, int], str] = ollama,
             log=sys.stderr) -> list[dict]:
    """Rate every item once. Depends on ``items`` and the pass config only.

    A batch that does not parse, or leaves a pair unscored, is retried one pair
    at a time; a pair that still fails is recorded as ``None``, never guessed.
    """
    swap, _, seed = PASSES[pass_no]
    order = pass_order(items, pass_no)
    results, started = [], time.time()
    for start in range(0, len(order), BATCH):
        chunk = order[start:start + BATCH]
        pairs = [(it["b"], it["a"]) if swap else (it["a"], it["b"]) for it in chunk]
        scores = parse_scores(ask(build_prompt(pairs), seed), len(pairs))
        for j, score in enumerate(scores):
            if score is None:
                scores[j] = parse_scores(ask(build_prompt([pairs[j]]), seed), 1)[0]
        for position, (it, score) in enumerate(zip(chunk, scores)):
            results.append({"id": it["id"], "score": score, "pass": pass_no,
                            "batch": start // BATCH, "position": position})
        if log and (start // BATCH) % 10 == 0:
            done = start + len(chunk)
            print(f"pass {pass_no}: {done}/{len(order)}  {time.time() - started:.0f}s",
                  file=log, flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="passes", type=int, nargs="*", default=list(PASSES))
    args = ap.parse_args()
    items = load_items()
    for pass_no in args.passes:
        rows = run_pass(items, pass_no)
        out = DATA / f"labels_pass{pass_no}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        missing = sum(r["score"] is None for r in rows)
        print(f"pass {pass_no}: wrote {len(rows)} rows, {missing} unscored -> {out}")
    meta = {"model": MODEL, "batch": BATCH, "temperature": 0,
            "passes": {k: {"swap": s, "shuffle_seed": sh, "sampling_seed": sd}
                       for k, (s, sh, sd) in PASSES.items()}}
    (DATA / "labels_config.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
