# Verification disagreements

Records the verifier judged to be in the **wrong family**. These are not bad sentences to discard -- they are claims that a generator is mislabelling its output, and a cluster of them is a bug with a return address.

- verifier: `ollama/qwen2.5:7b`
- records judged on category: 21 of 21
- judged miscategorised: **0** (0.0% of judged)
- distinct (labelled -> suggested) clusters: 0

> The verifier is a 7B local model and is itself imperfect -- see `verifier_validation.md` for how far it should be trusted. Treat a cluster as a lead to investigate, not a proven defect.

No miscategorisations were reported.
