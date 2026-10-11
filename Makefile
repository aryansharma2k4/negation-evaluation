# One entry point; every target is one line of real commands. What each script does: scripts/README.md.
PY := ./venv/bin/python
ROUNDS := composite f1 negation_detection negation_direction nli2 pattern polarity

.PHONY: setup data train eval test dashboard

# venv + pinned dependencies (spaCy model is pinned in requirements.txt) + WordNet
setup:
	python3 -m venv venv && ./venv/bin/pip install -r requirements.txt -r requirements-experiments.txt --extra-index-url https://download.pytorch.org/whl/cpu && $(PY) -c "import nltk; nltk.download('wordnet')"

# fetch MNLI/SNLI (pinned, sha256-verified), rebuild clean DEV, verify every dataset file against datasets/SHA256SUMS.
# Needs datasets/raw/negation_detection/ (CANNOT, STS-B, SNLI validation, xNot360) already present: no script fetches it.
data:
	$(PY) scripts/f2/fetch_data.py && $(PY) scripts/nli2/build_dev_clean.py --hard && cd datasets && sha256sum -c SHA256SUMS

# fit the adopted combiner (reports/09: 4-model GBM + overrides); a no-op when .cache/f1/detector_<key>.pkl exists
train:
	$(PY) -c "import sys; sys.path.insert(0, 'scripts/f1'); from detector import Detector; d = Detector.adopted(); print(d.cfg, d.thr)"

# reproduce the adopted configuration's headline numbers (reports/09 §1). Reads TEST: appends to reports/results/f2/test_reads.json
eval:
	$(PY) scripts/f2/final_f2.py

# the full suite: tests/, then each round separately (two rounds both have a module named data.py)
test:
	$(PY) -m pytest tests -q && for r in $(ROUNDS); do $(PY) -m pytest scripts/$$r -q || exit 1; done

dashboard:
	./venv/bin/streamlit run dashboard/app.py
