"""Phase 5: fine-tune a small cross-encoder on generator output ("is B the negation of A").

    ./venv/bin/python experiments/nli2/phase5.py --step data    # 5.1 regenerate training pairs (fixed generator)
    ./venv/bin/python experiments/nli2/phase5.py --step train   # 5.1/5.2 fine-tune, base-held-out check
    ./venv/bin/python experiments/nli2/phase5.py --step eval    # 5.3 alone / in cascade / off-the-shelf, TEST

Bases: the CANNOT phase-1 A sentences of the clean DEV set. None is in VALIDATION or TEST, so the
generator never sees a TEST base sentence. (negation_detection's refsets were generated from
phase-2 and xNot360 sentences too, and from the pre-0.1 affixal generator, so they are not used.)

Labels: the refsets rule (experiments/negation_detection/refsets.py): a variant is a negation
iff the parity of its net_negation differs from A's own cue parity. K1 cancellations and K2
compounds are label 0 by construction. Random re-pairings of two different DEV bases are added
as label 0, so "B differs from A" alone cannot be the learned rule.

Base sentences, not pairs, are held out: bases are split 90 / 10 by seed 0, and every pair of a
held-out base (variants and random pairs whose A is that base) is held out with it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import numpy as np

from common import CACHE, REPO, RESULTS, SEED, save
from common import read_dev_clean

TRAIN_JSONL = CACHE / "ft_pairs.jsonl"
MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
#: sha256 of the model.safetensors used (the hub blob name); checked at load
BASE_SHA256 = "ec9c5b6a08b60d7a01b465c7ed44b496244dfd1bed3274c95875a74cea89ab35"
MODEL_DIR = CACHE / "ft_model"
MAX_VARIANTS_PER_BASE = 8          # CPU budget: a few variants per base, families sampled evenly
BATCH = 200


def bases() -> list[str]:
    return sorted({r["a"] for r in read_dev_clean() if r["dataset"] == "cannot_phase1"})


def step_data() -> None:
    from negation import generate_all
    src = bases()
    rng = random.Random(SEED)
    pairs, stats = [], Counter()
    start = time.time()
    by_base: dict[str, list] = {}
    for i in range(0, len(src), BATCH):
        chunk = src[i:i + BATCH]
        try:
            records = generate_all(chunk)
        except Exception:
            records = []
            for s in chunk:
                try:
                    records.extend(generate_all([s]))
                except Exception:
                    stats["generator_error"] += 1
        for r in records:
            by_base.setdefault(r.base_sentence, []).append(r)
        print(f"  generated {min(i + BATCH, len(src))}/{len(src)} bases, {time.time() - start:.0f}s", flush=True)
    for a in src:
        rows = by_base.get(a, [])
        stats["variants_generated"] += len(rows)
        if not rows:
            stats["bases_without_variants"] += 1
            continue
        # sample evenly across families so C/H/I (most of the generator's output) do not dominate
        fams: dict[str, list] = {}
        for r in rows:
            fams.setdefault(r.family, []).append(r)
        for f in fams.values():
            rng.shuffle(f)
        picked = []
        while len(picked) < MAX_VARIANTS_PER_BASE and any(fams.values()):
            for f in sorted(fams):
                if fams[f] and len(picked) < MAX_VARIANTS_PER_BASE:
                    picked.append(fams[f].pop())
        for r in picked:
            label = int((r.net_negation % 2) != (r.input_cue_count % 2))
            pairs.append({"a": a, "b": r.variant, "label": label, "family": r.family, "subtype": r.subtype,
                          "source": "generator"})
    n_pos = sum(p["label"] for p in pairs)
    n_rand = max(0, n_pos - (len(pairs) - n_pos))            # top negatives up to a 50 / 50 balance
    for _ in range(n_rand):
        a, b = rng.sample(src, 2)
        pairs.append({"a": a, "b": b, "label": 0, "family": "random_pair", "subtype": "random_pair",
                      "source": "random"})
    rng.shuffle(pairs)
    CACHE.mkdir(exist_ok=True)
    TRAIN_JSONL.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    comp = Counter(f"{p['family']}|{p['label']}" for p in pairs)
    res = {"bases": len(src), "pairs": len(pairs), "labels": dict(Counter(str(p["label"]) for p in pairs)),
           "by_family_label": dict(sorted(comp.items())), "stats": dict(stats),
           "max_variants_per_base": MAX_VARIANTS_PER_BASE, "generator_commit_note": "after fix 0.1"}
    save("5_1_ft_data.json", res)
    print(json.dumps(res, indent=1))


EPOCHS = 1
LR = 2e-5
BATCH_SIZE = 16
MAX_LEN = 128
SCORES = CACHE / "ft_scores.pkl"


def split_bases(pairs) -> tuple[list[dict], list[dict], set[str]]:
    """90 / 10 by base sentence (seed 0). A pair is held out iff its A is a held-out base; a random
    pair whose two sentences fall on different sides is dropped, so no sentence crosses."""
    src = sorted({p["a"] for p in pairs})
    rng = random.Random(SEED)
    rng.shuffle(src)
    held = set(src[: len(src) // 10])
    train, test = [], []
    for p in pairs:
        side_a = p["a"] in held
        if p["source"] == "random" and (p["b"] in held) != side_a:
            continue
        (test if side_a else train).append(p)
    assert not ({p["a"] for p in train} | {p["b"] for p in train if p["source"] == "random"}) & held
    return train, test, held


def load_model(path=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if path is not None:
        return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path)
    import hashlib
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(MODEL_NAME, "model.safetensors")
    assert hashlib.sha256(open(weights, "rb").read()).hexdigest() == BASE_SHA256, "unexpected base weights"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    return tok, model


def predict(tok, model, pairs: list[tuple[str, str]], batch: int = 64) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            enc = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out.append(model(**enc).logits.softmax(-1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)


def step_train() -> None:
    import pickle
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    pairs = [json.loads(l) for l in TRAIN_JSONL.read_text().splitlines() if l.strip()]
    train, held, held_bases = split_bases(pairs)
    tok, model = load_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(train) + BATCH_SIZE - 1) // BATCH_SIZE)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = random.Random(SEED)
    log, start, step = [], time.time(), 0
    model.train()
    for epoch in range(EPOCHS):
        order = list(range(len(train)))
        rng.shuffle(order)
        for i in range(0, len(order), BATCH_SIZE):
            chunk = [train[j] for j in order[i:i + BATCH_SIZE]]
            enc = tok([p["a"] for p in chunk], [p["b"] for p in chunk], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            loss = model(**enc, labels=torch.tensor([p["label"] for p in chunk])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                log.append({"step": step, "loss": round(float(loss), 4), "seconds": round(time.time() - start)})
                print(f"  step {step}/{steps} loss {float(loss):.4f} {time.time() - start:.0f}s", flush=True)
    train_seconds = round(time.time() - start)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)

    p_held = predict(tok, model, [(p["a"], p["b"]) for p in held])
    y_held = np.array([p["label"] for p in held])
    by_fam = {}
    for fam in sorted({p["family"] for p in held}):
        idx = [i for i, p in enumerate(held) if p["family"] == fam]
        by_fam[fam] = {"n": len(idx), "accuracy_at_0.5": round(float(((p_held[idx] >= 0.5) == y_held[idx]).mean()), 4)}

    # score every evaluation pair once (DEV, VALIDATION, TEST, K1); TEST is only *used* in --step eval
    import pipeline as PL
    from phase4 import k1_rows
    splits, _, sl = PL.load()
    eval_pairs = list(dict.fromkeys((r["a"], r["b"]) for rows in (sl["dev_clean"], sl["validation"], sl["test"], k1_rows())
                                    for r in rows))
    p_eval = predict(tok, model, eval_pairs)
    SCORES.write_bytes(pickle.dumps(dict(zip(eval_pairs, p_eval.tolist()))))
    res = {"model": MODEL_NAME, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "max_len": MAX_LEN,
           "seed": SEED, "train_pairs": len(train), "heldout_pairs": len(held), "heldout_bases": len(held_bases),
           "train_seconds": train_seconds, "device": "cpu", "torch_threads": torch.get_num_threads(),
           "loss_log": log,
           "heldout_generator": {"accuracy_at_0.5": round(float(((p_held >= 0.5) == y_held).mean()), 4),
                                 "auc": round(float(roc_auc_score(y_held, p_held)), 4), "by_family": by_fam}}
    save("5_2_ft_train.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "loss_log"}, indent=1))


def step_eval() -> None:
    """5.3: fine-tuned alone, fine-tuned inside the cascade, off-the-shelf alone and in the cascade."""
    import pickle
    import pipeline as PL
    from common import C, compare, evaluate, log_test_read, youden_threshold
    from phase4 import adopted_cascade, k1_rows
    ft = pickle.loads(SCORES.read_bytes())
    splits, analyses, sl = PL.load()
    sl = dict(sl, k1=k1_rows())
    swap = lambda rows: [dict(r, nli_c=ft[(r["a"], r["b"])]) for r in rows]
    sl_ft = {k: swap(v) for k, v in sl.items()}
    log_test_read("5.3", "fine-tuned cross-encoder alone and in cascade vs off-the-shelf")
    systems = {}
    for tag, slices in (("off-the-shelf NLI (deberta-v3-base)", sl), ("fine-tuned MiniLM", sl_ft)):
        thr = youden_threshold(slices["dev_clean"], [r["nli_c"] for r in slices["dev_clean"]])
        sc = {k: np.array([r["nli_c"] for r in v]) for k, v in slices.items()}
        systems[f"{tag} alone"] = (sc, thr, slices)
        casc = PL.run(adopted_cascade(), splits, analyses, slices)
        systems[f"{tag} in cascade"] = (casc["_scores"], casc["stage3_threshold"], slices)
    res = {"rows": {}, "gains": {}}
    for name, (sc, thr, slices) in systems.items():
        ev = evaluate(slices["test"], sc["test"], thr)
        ev["validation_accuracy"] = evaluate(slices["validation"], sc["validation"], thr, parts=("all",))["all"]["accuracy"]
        ev["dev_clean_accuracy"] = evaluate(slices["dev_clean"], sc["dev_clean"], thr, parts=("all",))["all"]["accuracy"]
        ev["k1_fpr"] = round(float((sc["k1"] >= thr).mean()), 4)
        ev["semantoneg_phase2_fpr"] = round(float((sc["semantoneg_phase2"] >= thr).mean()), 4)
        res["rows"][name] = ev
    t = sl["test"]
    for a, b in (("off-the-shelf NLI (deberta-v3-base) alone", "fine-tuned MiniLM alone"),
                 ("off-the-shelf NLI (deberta-v3-base) in cascade", "fine-tuned MiniLM in cascade")):
        for part in ("all", "cannot_phase2", "xnot360"):
            res["gains"][f"{b} vs {a}|{part}"] = compare(t, (systems[a][0]["test"], systems[a][1]),
                                                         (systems[b][0]["test"], systems[b][1]), part)
    save("5_3_ft_eval.json", res)
    pc = lambda x: f"{100 * x:.1f}%"
    print("| system | phase-2 acc (95% CI) | xNot360 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc | clean DEV acc |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, e in res["rows"].items():
        f = e["fpr_by_kind"]
        ci = lambda part: f"{pc(e[part]['accuracy'])} ({pc(e[part]['ci95'][0])}–{pc(e[part]['ci95'][1])})"
        print(f"| {name} | {ci('cannot_phase2')} | {ci('xnot360')} | {ci('all')} | {e['all']['auc']:.3f} | {pc(e['all']['recall'])} | "
              + " | ".join(pc(f[k]["fpr"]) for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair", "xnot360_not_negation"))
              + f" | {pc(e['recall_by_type']['antonym']['recall'])} | {pc(e['k1_fpr'])} | {pc(e['semantoneg_phase2_fpr'])} | "
                f"{pc(e['validation_accuracy'])} | {pc(e['dev_clean_accuracy'])} |")
    for k, v in res["gains"].items():
        print(k, v["gain"], v["p_permutation"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["data", "train", "eval"])
    args = ap.parse_args()
    {"data": step_data, "train": step_train, "eval": step_eval}[args.step]()


if __name__ == "__main__":
    main()
