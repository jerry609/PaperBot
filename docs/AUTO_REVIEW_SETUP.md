# Auto Review Setup (CodeRabbit + Copilot + Gemini)

This repository now includes:

- `.coderabbit.yaml` to enable CodeRabbit auto review.
- `.github/CODEOWNERS` to auto-request code owners on PRs.
- `.github/workflows/gemini-review.yml` to run Gemini review on new/updated PRs.

## One-time GitHub settings

1. Install/enable GitHub Apps for this repo:
   - CodeRabbit
   - Gemini Assistant (or Gemini CLI workflow with `GEMINI_API_KEY`)
   - GitHub Copilot code review

2. Add repository secret:
   - `GEMINI_API_KEY`

3. (Recommended) Enable branch ruleset for PR quality gates:
   - Require pull request before merging
   - Require at least 1 approval
   - Require conversation resolution
   - Enable Copilot code review rule
   - Enable Code scanning / CI checks as needed

## Notes

- Gemini workflow skips automatically when `GEMINI_API_KEY` is missing.
- For fork PRs, secret access is restricted by GitHub; Gemini review may be skipped.
