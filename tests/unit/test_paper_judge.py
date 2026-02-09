import json

from paperbot.application.workflows.analysis.paper_judge import PaperJudge


class _FakeLLMService:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, **kwargs):
        return json.dumps(self.payload)

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "judge-model", "cost_tier": 2}


def test_paper_judge_single_parses_scores_and_overall():
    payload = {
        "relevance": {"score": 5, "rationale": "direct"},
        "novelty": {"score": 4, "rationale": "new"},
        "rigor": {"score": 4, "rationale": "solid"},
        "impact": {"score": 3, "rationale": "good"},
        "clarity": {"score": 5, "rationale": "clear"},
        "overall": 4.2,
        "one_line_summary": "strong paper",
        "recommendation": "must_read",
    }
    judge = PaperJudge(llm_service=_FakeLLMService(payload))

    result = judge.judge_single(paper={"title": "x", "snippet": "y"}, query="icl compression")

    assert result.relevance.score == 5
    assert result.overall == 4.2
    assert result.recommendation == "must_read"
    assert result.judge_model == "judge-model"


def test_paper_judge_calibration_uses_median():
    class _SwitchingLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            score = 3 if self.calls == 1 else (5 if self.calls == 2 else 4)
            payload = {
                "relevance": {"score": score, "rationale": ""},
                "novelty": {"score": 4, "rationale": ""},
                "rigor": {"score": 4, "rationale": ""},
                "impact": {"score": 4, "rationale": ""},
                "clarity": {"score": 4, "rationale": ""},
                "one_line_summary": "x",
                "recommendation": "worth_reading",
            }
            return json.dumps(payload)

        def describe_task_provider(self, task_type="default"):
            return {"provider_name": "fake", "model_name": "judge-model", "cost_tier": 1}

    judge = PaperJudge(llm_service=_SwitchingLLM())
    result = judge.judge_with_calibration(paper={"title": "x", "snippet": "y"}, query="icl", n_runs=3)

    assert result.relevance.score == 4
    assert result.recommendation in {"must_read", "worth_reading", "skim", "skip"}
