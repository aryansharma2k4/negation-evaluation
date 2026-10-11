"""Phase 5.1: 500 OS / CN / DBMS definitions from a real source (English Wikipedia lead sentences).

    ./venv/bin/python experiments/f1/fetch_definitions.py   -> data/definitions.jsonl, results/5_1_definitions.json

For each subject, the members (articles, namespace 0) of fixed concept categories are listed; each
article's first lead sentence comes from the TextExtracts API (exintro, explaintext, exsentences=1),
with the article's revision id at fetch time. A sentence is kept only if it is definition-shaped:
"<term> is / are / refers to / describes ... ", 8-45 words, not a disambiguation page. Parenthetical
asides "( ... )" are removed (pronunciations, abbreviations, dates) and recorded as such; nothing else
in the text is changed. Then 167 / 167 / 166 are sampled per subject (seed 0). Wikipedia text is
CC BY-SA 4.0; every row carries its source URL and revision id for attribution.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import urllib.request

from core import REPO, SEED, save

API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "negation-research/0.1 (academic project; contact via repository)"}
SUBJECTS = {   # order matters: an article in two subjects' categories goes to the first
    "DBMS": ["Database_theory", "Relational_model", "Transaction_processing", "SQL", "Database_algorithms",
             "Concurrency_control_algorithms", "Database_index_techniques"],
    "OS": ["Operating_system_technology", "Process_(computing)", "Memory_management", "Processor_scheduling_algorithms",
           "Scheduling_(computing)", "Computer_file_systems", "Virtual_memory", "Inter-process_communication",
           "Operating_system_kernels", "Concurrency_control"],
    "CN": ["Routing", "Transport_layer_protocols", "Network_layer_protocols", "Link_protocols", "Network_topology",
           "Error_detection_and_correction", "Network_architecture", "Internet_protocols", "Network_protocols"],
}
TARGET = {"OS": 167, "CN": 167, "DBMS": 166}
#: Company / product / venue pages are not concept definitions a teacher would set.
PRODUCT_RE = re.compile(r"\b(company|corporation|Inc\.|Ltd|proprietary|manufactur\w*|developed by|founded|headquartered|"
                        r"conference|journal|software product|brand)\b", re.I)
DEF_RE = re.compile(r"^.{2,120}?\b(is|are|was|were|refers to|describes|denotes|consists of)\b", re.I)


def _api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(8):
        time.sleep(0.5)                              # be polite: the API rate-limits bursts
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:
            print("  retry", attempt, repr(e)[:120], flush=True)
            time.sleep(5 * (attempt + 1))
    raise IOError(url)


def members(cat: str) -> list[str]:
    out, cont = [], {}
    while True:
        d = _api({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmnamespace": 0,
                  "cmlimit": 500, **cont})
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return out
        cont = d["continue"]


def leads(titles: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(titles), 20):
        d = _api({"action": "query", "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exsentences": 1,
                  "redirects": 1, "titles": "|".join(titles[i:i + 20])})
        for p in d["query"]["pages"]:
            if "extract" in p:
                out[p["title"]] = {"raw": p["extract"].strip(), "revid": p.get("lastrevid"), "pageid": p.get("pageid")}
    return out


def clean(raw: str) -> str | None:
    s = raw.split("\n")[0].strip()
    prev = None
    while prev != s:                                  # nested parentheses
        prev, s = s, re.sub(r"\s*\([^()]*\)", "", s)
    s = re.sub(r"\s+", " ", s).replace(" ,", ",").strip()
    n = len(s.split())
    if not (8 <= n <= 45) or "may refer to" in s or not s.endswith(".") or not DEF_RE.match(s) or PRODUCT_RE.search(s):
        return None
    return s


def main() -> None:
    seen, pools, stats = set(), {}, {}
    for subj, cats in SUBJECTS.items():
        titles = []
        for c in cats:
            titles += [t for t in members(c) if t not in seen and not t.startswith(("List of", "Comparison of", "Outline of"))]
        titles = list(dict.fromkeys(titles))
        seen |= set(titles)
        got = leads(titles)
        rows = []
        for t, info in got.items():
            text = clean(info["raw"])
            if text and not re.search(r"\b(19|20)\d\d\b", t):
                rows.append({"subject": subj, "term": t, "definition": text, "source": "wikipedia",
                             "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                             "revid": info["revid"], "raw_lead_sentence": info["raw"].split("\n")[0].strip(),
                             "edited": "parentheticals removed" if text != info["raw"].split("\n")[0].strip() else "none",
                             "authorship": "real source (Wikipedia editors), not written for this project"})
        pools[subj] = sorted(rows, key=lambda r: r["term"])
        stats[subj] = {"categories": cats, "articles": len(titles), "with_lead": len(got), "definition_shaped": len(rows)}
        print(subj, stats[subj], flush=True)
    rng = random.Random(SEED)
    final = []
    for subj, n in TARGET.items():
        take = rng.sample(pools[subj], min(n, len(pools[subj])))
        final += sorted(take, key=lambda r: r["term"])
        stats[subj]["sampled"] = len(take)
    for i, r in enumerate(final):
        r["id"] = f"def{i:03d}"
    (REPO / "data").mkdir(exist_ok=True)
    with (REPO / "data" / "definitions.jsonl").open("w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    save("5_1_definitions.json", {"source": "English Wikipedia, TextExtracts API (first lead sentence), fetched "
                                            + time.strftime("%Y-%m-%d"), "licence": "CC BY-SA 4.0 (attribution: url + revid per row)",
                                  "filters": "definition-shaped (is/are/refers to ...), 8-45 words, not disambiguation, not a "
                                             "company / product / venue page (PRODUCT_RE), no year in the title; "
                                             "parentheticals removed", "per_subject": stats, "total": len(final),
                                  "self_written": 0})
    print(len(final), "definitions")


if __name__ == "__main__":
    main()
