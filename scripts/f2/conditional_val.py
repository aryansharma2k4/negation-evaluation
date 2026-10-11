"""Phase 5: a synthetic conditional validation set, and variant B of reports/07-honest-baseline.md step 3.3 validated off TEST.

    ./venv/bin/python scripts/f2/conditional_val.py --step 5.1   # build; append to datasets/splits/val_ood.csv
    ./venv/bin/python scripts/f2/conditional_val.py --step 5.2   # variant B on this set and clean DEV, then TEST

All pairs are **machine-generated** from templates over 44 hand-written content tuples (P, not-P, Q,
not-Q). Labels follow from the logic of the template, never from annotation:

  row                     A               B                    label  derivation
  antecedent flip         If P, Q.        If not P, Q.          0     both can hold (Q whatever P is)
  inverse                 If P, Q.        If not P, not Q.      0     both can hold (Q iff P); B is not A's negation
  consequent flip         If P, Q.        If P, not Q.          1     cannot both hold whenever P is possible
  logical negation        If P, Q.        P, but not Q.         1     B is exactly not(P -> Q) = P and not Q
  contrapositive          If P, Q.        If not Q, not P.      0     logically equivalent to A
  unless                  If P, Q.        Unless P, Q.          0     'unless P' = 'if not P': an antecedent flip
  even if                 If P, Q.        Even if not P, Q.     0     asserts Q regardless of P; entails A
  only if                 If P, Q.        Q only if P.          0     = if Q then P (the converse); compatible with A
Rows 1-5 also appear postposed (Q if P. / Q if not P. ...). Labels follow CANNOT / NLI semantics: a
negation is a B that cannot be true together with A.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict

import numpy as np

from core2 import C, REPO, STOPPED, load, log_test_read, save

import val_ood as VO

# (topic, P, not-P, Q, not-Q): clauses, lower-case initial unless a proper noun / acronym
CONTENT = [
    ("OS", "a process holds a lock", "a process does not hold a lock", "other processes wait", "other processes do not wait"),
    ("OS", "the ready queue is empty", "the ready queue is not empty", "the CPU stays idle", "the CPU does not stay idle"),
    ("OS", "a page is not in memory", "a page is in memory", "a page fault occurs", "no page fault occurs"),
    ("OS", "the time quantum is very large", "the time quantum is not very large", "round robin behaves like FCFS", "round robin does not behave like FCFS"),
    ("OS", "a thread blocks on I/O", "a thread does not block on I/O", "the scheduler picks another thread", "the scheduler does not pick another thread"),
    ("OS", "all four deadlock conditions hold", "not all four deadlock conditions hold", "a deadlock can occur", "a deadlock cannot occur"),
    ("OS", "the system is in a safe state", "the system is not in a safe state", "the request is granted", "the request is not granted"),
    ("OS", "the TLB has the entry", "the TLB does not have the entry", "the page table is not consulted", "the page table is consulted"),
    ("OS", "a process exceeds its quantum", "a process does not exceed its quantum", "it is preempted", "it is not preempted"),
    ("OS", "the working set fits in memory", "the working set does not fit in memory", "thrashing is avoided", "thrashing is not avoided"),
    ("OS", "a semaphore value is zero", "a semaphore value is not zero", "the wait call blocks", "the wait call does not block"),
    ("CN", "a packet is lost", "a packet is not lost", "TCP retransmits it", "TCP does not retransmit it"),
    ("CN", "the receive window is full", "the receive window is not full", "the sender stops sending", "the sender does not stop sending"),
    ("CN", "the checksum fails", "the checksum does not fail", "the segment is discarded", "the segment is not discarded"),
    ("CN", "the TTL reaches zero", "the TTL does not reach zero", "the router drops the packet", "the router does not drop the packet"),
    ("CN", "the DNS cache has the record", "the DNS cache does not have the record", "no query is sent upstream", "a query is sent upstream"),
    ("CN", "a collision is detected", "no collision is detected", "the station backs off", "the station does not back off"),
    ("CN", "the link is congested", "the link is not congested", "the congestion window shrinks", "the congestion window does not shrink"),
    ("CN", "the switch knows the MAC address", "the switch does not know the MAC address", "the frame is forwarded on one port", "the frame is not forwarded on one port"),
    ("CN", "the ACK arrives late", "the ACK does not arrive late", "the timer expires", "the timer does not expire"),
    ("CN", "the route is unreachable", "the route is not unreachable", "an ICMP message is returned", "no ICMP message is returned"),
    ("CN", "the handshake completes", "the handshake does not complete", "the connection is established", "the connection is not established"),
    ("DBMS", "a transaction commits", "a transaction does not commit", "its changes are durable", "its changes are not durable"),
    ("DBMS", "a transaction aborts", "a transaction does not abort", "its changes are rolled back", "its changes are not rolled back"),
    ("DBMS", "the index covers the query", "the index does not cover the query", "the table is not scanned", "the table is scanned"),
    ("DBMS", "a lock is held exclusively", "a lock is not held exclusively", "other transactions cannot read the row", "other transactions can read the row"),
    ("DBMS", "the schedule is conflict serializable", "the schedule is not conflict serializable", "it is serializable", "it is not serializable"),
    ("DBMS", "a foreign key is violated", "a foreign key is not violated", "the insert is rejected", "the insert is not rejected"),
    ("DBMS", "the relation is in 3NF", "the relation is not in 3NF", "it has no transitive dependencies", "it has transitive dependencies"),
    ("DBMS", "the buffer pool is full", "the buffer pool is not full", "a page is evicted", "no page is evicted"),
    ("DBMS", "the log is flushed", "the log is not flushed", "the commit is acknowledged", "the commit is not acknowledged"),
    ("DBMS", "two transactions update the same row", "two transactions do not update the same row", "one of them waits", "neither of them waits"),
    ("DBMS", "the query uses a join", "the query does not use a join", "the optimizer chooses a join order", "the optimizer does not choose a join order"),
    ("everyday", "it rains", "it does not rain", "the match is cancelled", "the match is not cancelled"),
    ("everyday", "you follow the traffic signals", "you do not follow the traffic signals", "you avoid accidents", "you do not avoid accidents"),
    ("everyday", "the shop is open", "the shop is not open", "we buy bread", "we do not buy bread"),
    ("everyday", "you water the plant", "you do not water the plant", "it grows", "it does not grow"),
    ("everyday", "the train is late", "the train is not late", "she misses the meeting", "she does not miss the meeting"),
    ("everyday", "the alarm rings", "the alarm does not ring", "he wakes up", "he does not wake up"),
    ("everyday", "you pay the fee", "you do not pay the fee", "you receive the certificate", "you do not receive the certificate"),
    ("everyday", "the milk is cold", "the milk is not cold", "the children drink it", "the children do not drink it"),
    ("everyday", "the road is icy", "the road is not icy", "the bus runs slowly", "the bus does not run slowly"),
    ("everyday", "the door is locked", "the door is not locked", "the guests wait outside", "the guests do not wait outside"),
    ("everyday", "the battery is charged", "the battery is not charged", "the phone turns on", "the phone does not turn on"),
]


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def pairs() -> list[dict]:
    out = []
    for i, (topic, p, np_, q, nq) in enumerate(CONTENT):
        A = f"If {p}, {q}."
        rows = [("antecedent flip", "preposed", f"If {np_}, {q}.", 0),
                ("inverse", "preposed", f"If {np_}, {nq}.", 0),
                ("consequent flip", "preposed", f"If {p}, {nq}.", 1),
                ("logical negation", "fact", f"{cap(p)}, but {nq}.", 1),
                ("contrapositive", "preposed", f"If {nq}, {np_}.", 0),
                ("unless", "preposed", f"Unless {p}, {q}.", 0),
                ("even if", "preposed", f"Even if {np_}, {q}.", 0),
                ("only if", "postposed", f"{cap(q)} only if {p}.", 0)]
        A_post = f"{cap(q)} if {p}."
        rows += [("antecedent flip", "postposed", f"{cap(q)} if {np_}.", 0),
                 ("inverse", "postposed", f"{cap(nq)} if {np_}.", 0),
                 ("consequent flip", "postposed", f"{cap(nq)} if {p}.", 1),
                 ("contrapositive", "postposed", f"{cap(np_)} if {nq}.", 0)]
        for tmpl, order, b, lab in rows:
            a = A_post if order == "postposed" and tmpl != "only if" else A
            out.append({"a": a, "b": b, "label": lab, "dataset": "val_ood_conditionals", "kind": f"cond_{tmpl.replace(' ', '_')}",
                        "neg_type": "", "source": f"synthetic template, content tuple {i} ({topic})", "template": f"{tmpl}|{order}"})
    return out


def step_5_1() -> None:
    cond = pairs()
    VO.assert_disjoint(cond)
    old = [r for r in VO.read() if r["dataset"] != "val_ood_conditionals"]
    VO.write(old + cond)
    rows = VO.read()
    c = [r for r in rows if r["dataset"] == "val_ood_conditionals"]
    res = {"content_tuples": len(CONTENT), "topics": dict(Counter(t[0] for t in CONTENT)), "pairs": len(c),
           "by_template": dict(Counter(r["template"] for r in c)), "labels": dict(Counter(r["label"] for r in c)),
           "halves": dict(Counter(r["part"] for r in c)), "val_ood_total": len(rows),
           "examples": [{k: r[k] for k in ("a", "b", "label", "template")} for r in c[:12]]}
    save("5_1_conditionals.json", res)
    print(res)


def variant_b(rows) -> np.ndarray:
    """Report_NLI2 3.3 variant B: both sides conditional and the antecedent-only flip or both halves flipped
    (inverse / contrapositive) -> not a negation (veto only; never forces a negation)."""
    import conditionals as K
    sents = list(dict.fromkeys(s for r in rows for s in (r["a"], r["b"])))
    conds = K.conditionals(sents)
    an = C.P.analyse_many(sents)
    return np.array([K.decide(K.primary(conds[r["a"]]), K.primary(conds[r["b"]]), an[r["a"]], an[r["b"]]) in ("not_negation", "contraposition")
                     for r in rows])


def step_5_2() -> None:
    import deberta as D
    import eval2 as E2
    import ensemble as E
    ctx = D.context()           # rebuilt: val_ood now includes the conditionals
    X_c = {k: np.hstack([ctx["X"][k]] + [ctx["P"][m][k] for m in D.PRUNED if m != "deberta_base"]) for k in E2.SETS}
    g = E.fit_gbm(ctx["sets"], X_c["dev_clean"])
    sc = {k: np.where(ctx["masks"][k], STOPPED, g["model"].predict_proba(X_c[k])[:, 1]) for k in E2.SETS}
    cur = D.with_rule(ctx, sc, "deberta_base")
    veto = {k: variant_b(ctx["sets"][k]) for k in E2.SETS}
    new = {k: np.where(veto[k], STOPPED, cur[k]) for k in E2.SETS}
    thr = g["threshold"]
    # validation off TEST: the conditional set (all of it) and clean DEV's conditional slice
    ood = ctx["ood"]
    yo = C.labels(ood)
    per_row = defaultdict(dict)
    for name, s in (("current (C + override)", cur), ("current + variant B", new)):
        po = np.asarray(s["val_ood"]) >= thr
        for t in sorted({r["template"] for r in ood if r["dataset"] == "val_ood_conditionals"}):
            m = np.array([r["template"] == t for r in ood])
            per_row[t][name] = {"n": int(m.sum()), "accuracy": round(float((po[m] == yo[m]).mean()), 4)}
        m = np.array([r["dataset"] == "val_ood_conditionals" for r in ood])
        per_row["ALL conditionals"][name] = {"n": int(m.sum()), "accuracy": round(float((po[m] == yo[m]).mean()), 4)}
    import conditionals as K
    d = ctx["sets"]["dev_clean"]
    conds = K.conditionals([s for r in d for s in (r["a"], r["b"])])
    both = np.array([bool(K.primary(conds[r["a"]])) and bool(K.primary(conds[r["b"]])) for r in d])
    yd = C.labels(d)
    oof = np.where(ctx["masks"]["dev_clean"], STOPPED, g["oof"])
    oof_rule = D.with_rule(ctx, {**sc, "dev_clean": oof}, "deberta_base")["dev_clean"]
    dev_slice = {}
    for name, s in (("current (out-of-fold)", oof_rule), ("current + variant B (out-of-fold)", np.where(veto["dev_clean"], STOPPED, oof_rule))):
        p = s >= g["oof_thr"] if "oof_thr" in g else s >= thr
        dev_slice[name] = {"n": int(both.sum()), "accuracy": round(float((p[both] == yd[both]).mean()), 4),
                           "vetoes_fired": int((veto["dev_clean"] & both).sum()), "veto_on_real_negations": int((veto["dev_clean"] & both & (yd == 1)).sum())}
    base = E2.row(ctx["sets"], cur, thr, ood)
    after = E2.row(ctx["sets"], new, thr, ood)
    passes_offtest = per_row["ALL conditionals"]["current + variant B"]["accuracy"] > per_row["ALL conditionals"]["current (C + override)"]["accuracy"] \
        and dev_slice["current + variant B (out-of-fold)"]["accuracy"] >= dev_slice["current (out-of-fold)"]["accuracy"]
    adopt = passes_offtest and E2.adopt(base, after)
    res = {"per_template_row": per_row, "dev_conditional_slice": dev_slice, "passes_off_test": passes_offtest,
           "selection_rule_passed": E2.adopt(base, after), "adopted": adopt,
           "validation_and_val_ood": {"current": {k: base[k] for k in ("validation", "val_ood_select", "val_ood_by_dataset")},
                                      "with_variant_B": {k: after[k] for k in ("validation", "val_ood_select", "val_ood_by_dataset")}}}
    if adopt:
        log_test_read("5.2", "variant B on TEST (first honest measurement)")
        res["test"] = {"current (C + override)": base, "current + variant B": after,
                       "gain": __import__("override").F1gain(ctx, new, thr), "gain_vs_current": {
                           p: __import__("core2").N.compare(ctx["sets"]["test"], (cur["test"], thr), (new["test"], thr), p)
                           for p in ("all", "cannot_phase2", "xnot360")}}
        tst = ctx["sets"]["test"]
        tconds = K.conditionals([s for r in tst for s in (r["a"], r["b"])])
        inv = np.array([r["dataset"] == "xnot360" and bool(K.primary(tconds[r["a"]])) and bool(K.primary(tconds[r["b"]])) for r in tst])
        yt = C.labels(tst)
        res["test"]["xnot360_both_conditional"] = {n: round(float(((np.asarray(s["test"]) >= thr)[inv] == yt[inv]).mean()), 4)
                                                  for n, s in (("current", cur), ("variant B", new))}
        res["test"]["vetoes_on_test"] = {"fired": int(veto["test"].sum()), "on_real_negations": int((veto["test"] & (yt == 1)).sum())}
    save("5_2_variant_b.json", res)
    import json
    print(json.dumps({k: v for k, v in res.items() if k != "test"}, indent=1, default=str)[:5000])
    if adopt:
        print(E2.HEADER)
        print(E2.line("current (C + override)", base))
        print(E2.line("+ variant B", after))
        print(res["test"]["gain_vs_current"], res["test"]["xnot360_both_conditional"], res["test"]["vetoes_on_test"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["5.1", "5.2"])
    {"5.1": step_5_1, "5.2": step_5_2}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
