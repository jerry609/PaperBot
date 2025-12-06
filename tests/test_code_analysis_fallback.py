from agents.code_analysis_agent import CodeAnalysisAgent


def test_code_analysis_placeholder_no_repo():
    agent = CodeAnalysisAgent({})
    res = agent._placeholder(None, "no_repository_provided")
    assert res["placeholder"] is True
    assert res["repo_url"] is None
    assert res["stars"] is None


def test_code_analysis_process_without_repo_returns_placeholder():
    agent = CodeAnalysisAgent({})
    res = agent.process()
    assert res["placeholder"] is True

