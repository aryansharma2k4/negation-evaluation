"""The adopted system as one object that scores arbitrary (teacher, student) pairs, for Phases 4-6.

    det = Detector.adopted()            # configuration recorded in results/final_config.json
    out = det.predict([("T sentence", "S sentence"), ...])

Stages, in order (the dashboard shows which one decided):
    1. cancellation override (Phase 0.1): cancellation in either direction, or not + antonym -> "not a negation"
    2. combiner: GBM over the 17 features (+ the Phase 2 NLI ensemble features, if adopted); prob >= threshold
    3. guardrail (Phase 3, if adopted): downgrade a combiner "negation" whose margin <= delta
The combiner is fit on clean DEV exactly as in the phase scripts (deterministic, seed 0) and cached.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass

import numpy as np

from core import C, CACHE, RESULTS

import phase0 as P0
import phase4 as P4  # noqa: E402  (nli2)
import rules as R  # noqa: E402
import variants as V
from nli_models import scorer

REF = "deberta_base"


@dataclass(frozen=True)
class Config:
    nli_keys: tuple = (REF,)            # models whose probabilities the combiner sees
    method: str = "phase0"               # "phase0" | "gbm_all_probs" | "gbm_mean_probs"
    guardrail: dict | None = None        # {"sim": "cos", "agg": "max", "delta": float} or None
    rule: dict | None = None             # report_f2 1.3: {"theta", "rho", "kappa"}: low-overlap NLI override

    def key(self) -> str:
        d = asdict(self)
        d.pop("rule")          # the rule acts after the combiner, so it does not change the fitted model
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _probs(key: str, rows) -> np.ndarray:
    got = scorer(key).score([(r["a"], r["b"]) for r in rows])
    return np.array([[got[(r["a"], r["b"])][l] for l in ("contradiction", "entailment", "neutral")] for r in rows])


def extend(cfg: Config, X: np.ndarray, P: dict) -> np.ndarray:
    """Base 17 features -> the combiner's input for cfg. P: model key -> (n, 3) probabilities."""
    others = [k for k in cfg.nli_keys if k != REF]
    if cfg.method == "gbm_all_probs":
        return np.hstack([X] + [P[k] for k in others])
    if cfg.method == "gbm_mean_probs":
        X = X.copy()
        X[:, 1:4] = np.mean([P[k] for k in cfg.nli_keys], 0)
    return X


def feature_names(cfg: Config) -> list[str]:
    others = [k for k in cfg.nli_keys if k != REF]
    return P4.FEATURES + ([f"{k}:{l}" for k in others for l in ("p_c", "p_e", "p_n")] if cfg.method == "gbm_all_probs" else [])


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = CACHE / f"detector_{cfg.key()}.pkl"
        if path.exists():
            self.model, self.thr = pickle.loads(path.read_bytes())
            return
        import ensemble as P2
        b = P0.baseline()
        if cfg.method == "phase0":
            self.model, self.thr = b["fit"]["_model"], b["thr"]
        else:
            P = {k: P2.probs(k, b["sets"]["dev_clean"]) for k in cfg.nli_keys}
            g = P2.fit_gbm(b["sets"], extend(cfg, b["X"]["dev_clean"], P))
            self.model, self.thr = g["model"], g["threshold"]
        CACHE.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps((self.model, self.thr)))

    @classmethod
    def adopted(cls) -> "Detector":
        """The latest adopted configuration: report_f2's if it exists, else report_f1's."""
        f2 = RESULTS.parent / "f2" / "final_config.json"
        d = json.loads((f2 if f2.exists() else RESULTS / "final_config.json").read_text())
        return cls(Config(tuple(d["nli_keys"]), d["method"], d.get("guardrail"), d.get("rule")))

    def predict(self, pairs: list[tuple[str, str]], explain: bool = False) -> list[dict]:
        rows = [{"a": a, "b": b, "dataset": "new", "kind": "new", "label": 0} for a, b in pairs]
        sents = list(dict.fromkeys(s for r in rows for s in (r["a"], r["b"])))
        analyses = C.P.analyse_many(sents)
        emb = V.embed(sents)
        P = {k: _probs(k, rows) for k in dict.fromkeys((REF,) + tuple(self.cfg.nli_keys))}
        for r, p, in zip(rows, P[REF]):
            r["nli_c"], r["nli_e"], r["nli_n"] = map(float, p)
            r["cos"] = float(emb[r["a"]] @ emb[r["b"]])
        X = P4.featurise(rows, analyses)
        Xc = extend(self.cfg, X, P)
        prob = self.model.predict_proba(Xc)[:, 1]
        override = P0.override_mask(rows, analyses)
        g = self.cfg.guardrail
        out = []
        for i, r in enumerate(rows):
            stage, verdict = "combiner", bool(prob[i] >= self.thr)
            margin = info = None
            rule = self.cfg.rule
            if override[i]:
                stage, verdict = "cancellation override", False
            elif not verdict and rule and X[i][P4.FEATURES.index("token_overlap")] < rule["theta"] \
                    and r["nli_c"] > rule["rho"] and r["cos"] >= rule["kappa"]:
                stage, verdict = "low-overlap NLI override", True
            elif verdict and g:
                margin, info = self.margin(r["a"], r["b"], g)
                if margin is not None and margin <= g["delta"]:
                    stage, verdict = "guardrail downgrade", False
            o = {"teacher": r["a"], "student": r["b"], "negation": verdict, "prob": float(prob[i]),
                 "threshold": float(self.thr), "stage": stage, "cos": r["cos"]}
            if explain:
                if g and margin is None and not override[i]:
                    margin, info = self.margin(r["a"], r["b"], g)
                o.update({"features": dict(zip(feature_names(self.cfg), map(float, Xc[i]))),
                          "contributions": self.contributions(Xc[i]),
                          "nli": {k: dict(zip(("contradiction", "entailment", "neutral"), map(float, P[k][i]))) for k in P},
                          "override": bool(override[i]), "guardrail": {"margin": margin, **(info or {})} if g else None})
            out.append(o)
        return out

    def margin(self, t: str, s: str, g: dict):
        import guardrail as P3
        neg = V.negations([t])[t]
        if not neg:
            return None, {"variants": [], "note": "generator produced no negation of T; guardrail abstains"}
        emb = V.embed([t, s] + [v for v, _ in neg])
        sv = P3.sims(g["sim"], s, [v for v, _ in neg], emb)
        s_orig = float(P3.sims(g["sim"], s, [t], emb)[0])
        srt = np.sort(sv)[::-1]
        s_neg = float(srt[0] if g["agg"] == "max" else srt[:3].mean())
        ranked = sorted(zip(neg, sv), key=lambda x: -x[1])
        return s_neg - s_orig, {"s_orig": s_orig, "s_neg": s_neg, "delta": g["delta"],
                                "variants": [{"text": v, "family": f, "sim": float(x)} for (v, f), x in ranked]}

    def contributions(self, x: np.ndarray, top: int = 6) -> list[tuple[str, float]]:
        """Per-feature contribution to this decision: change in the raw GBM output when the feature
        is replaced by its clean-DEV median (a local, approximate attribution; not a calibrated quantity)."""
        med = self._medians()
        base = float(self.model.predict_proba(x[None])[0, 1])
        deltas = []
        for j, name in enumerate(feature_names(self.cfg)):
            z = x.copy()
            z[j] = med[j]
            deltas.append((name, base - float(self.model.predict_proba(z[None])[0, 1])))
        return sorted(deltas, key=lambda kv: -abs(kv[1]))[:top]

    def _medians(self) -> np.ndarray:
        if not hasattr(self, "_med"):
            path = CACHE / f"detector_{self.cfg.key()}_medians.npy"
            if path.exists():
                self._med = np.load(path)
            else:
                import ensemble as P2
                b = P0.baseline()
                P = {k: P2.probs(k, b["sets"]["dev_clean"]) for k in self.cfg.nli_keys}
                self._med = np.median(extend(self.cfg, b["X"]["dev_clean"], P), 0)
                np.save(path, self._med)
        return self._med


def scopes(sentence: str) -> list[dict]:
    """Every negation cue with its clause scope (token indices), for display."""
    from negation.nlp_core import clause_scope, parse_batch
    doc = parse_batch([sentence])[0]
    a = C.P.extract(doc)
    out = []
    for c in a.cues:
        head = doc[c.clause]
        idx = sorted(clause_scope(head))
        cue_i = next((t.i for t in doc if t.i in idx and (t.lower_ == c.text or c.text.startswith(t.lower_))), None)
        out.append({"cue": c.text, "family": c.family, "cue_index": cue_i, "scope": idx, "main_clause": c.main_clause,
                    "tokens": [t.text_with_ws for t in doc]})
    return out


def conditional(sentence: str) -> dict | None:
    import conditionals as K
    c = K.primary(K.conditionals([sentence])[sentence])
    if c is None:
        return None
    from negation.nlp_core import parse_batch
    doc = parse_batch([sentence])[0]
    a = C.P.extract(doc)
    pa, pc = K.polarities(c, a)
    words = lambda span: " ".join(doc[i].text for i in span)
    return {"marker": c.marker, "antecedent": words(c.ante_span), "consequent": words(c.cons_span),
            "antecedent_negated": bool(pa), "consequent_negated": bool(pc)}


if __name__ == "__main__":      # smoke check
    d = Detector(Config())
    for o in d.predict([("TCP is a reliable protocol.", "TCP is not a reliable protocol."),
                        ("TCP is a reliable protocol.", "TCP is a dependable protocol."),
                        ("Page faults are common.", "Page faults are not uncommon.")]):
        print(o["negation"], o["stage"], round(o["prob"], 3), o["teacher"], "/", o["student"])
    assert d.predict([("TCP is a reliable protocol.", "TCP is not a reliable protocol.")])[0]["negation"]
