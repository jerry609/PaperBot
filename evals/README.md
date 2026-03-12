# Evals (Smoke & Regression)

This directory is the home for PaperBot regression/evaluation cases.

Phase-0 intent:
- Provide a stable place to add smoke cases for the two main trunks:
  - `scholar_pipeline`
  - `paper2code`
- Add deterministic retrieval quality regression coverage for `PaperSearchService`
- Add deterministic document-evidence retrieval coverage for indexed paper chunks
- Track basic metrics (success rate, latency, cost estimate) and catch regressions early.

Suggested layout (see `docs/PLAN.md` for details):

```
evals/
  cases/
  fixtures/
  runners/
  scorers/
  reports/
```
