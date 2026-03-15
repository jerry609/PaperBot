from __future__ import annotations

import pytest

from paperbot.agents.code_analysis.agent import CodeAnalysisAgent
from paperbot.agents.quality.agent import QualityAgent


@pytest.mark.asyncio
async def test_single_repo_mode_preserves_placeholder_result():
    agent = CodeAnalysisAgent({})
    placeholder = agent._placeholder("https://github.com/example/repo", "clone_failed")

    async def fake_process_batch(_links):
        return {
            "repositories_analyzed": 1,
            "analysis_results": [placeholder],
        }

    agent._process_batch = fake_process_batch  # type: ignore[method-assign]

    result = await agent.process(repo_url="https://github.com/example/repo")

    assert result["placeholder"] is True
    assert result["reason"] == "clone_failed"
    assert result["repo_url"] == "https://github.com/example/repo"


def test_flatten_result_maps_code_meta_contract_fields():
    agent = CodeAnalysisAgent({})

    flattened = agent._flatten_result(
        {
            "repo_url": "https://github.com/example/repo",
            "analysis": {
                "structure_analysis": {
                    "primary_language": "Python",
                    "documentation": {"has_readme": True},
                },
                "quality_analysis": {
                    "overall_score": 0.82,
                    "recommendations": ["Add more tests"],
                },
            },
            "meta": {
                "stars": 12,
                "forks": 3,
                "updated_at": "2026-03-01T00:00:00+00:00",
                "last_commit_at": "2026-03-02T00:00:00+00:00",
            },
        }
    )

    assert flattened["updated_at"] == "2026-03-01T00:00:00+00:00"
    assert flattened["last_commit_date"] == "2026-03-02T00:00:00+00:00"
    assert flattened["reproducibility_score"] == 82.0
    assert flattened["has_readme"] is True


@pytest.mark.asyncio
async def test_quality_agent_accepts_flat_workflow_code_analysis_result():
    agent = QualityAgent({})

    result = await agent.process(
        code_analysis_result={
            "repo_url": "https://github.com/example/repo",
            "reproducibility_score": 72.0,
            "has_readme": True,
        }
    )

    assert result["quality_score"] == 0.72
    assert "README" in " ".join(result["strengths"])


@pytest.mark.asyncio
async def test_quality_agent_scores_nested_analysis_contract():
    agent = QualityAgent({})

    result = await agent.process(
        analysis_results={
            "analysis_results": [
                {
                    "repo_url": "https://github.com/example/repo",
                    "analysis": {
                        "structure_analysis": {
                            "files": {
                                "file_paths": [
                                    "src/app.py",
                                    "tests/test_app.py",
                                ]
                            },
                            "complexity": {
                                "overall_complexity": 4,
                                "file_complexity": {
                                    "src/app.py": {"total_complexity": 4}
                                },
                            },
                            "documentation": {
                                "docstring_coverage": 0.8,
                                "readme_quality": 0.9,
                                "api_documentation": {"coverage": 0.75},
                            },
                        },
                        "quality_analysis": {
                            "overall_score": 0.8,
                            "complexity_score": 0.9,
                            "maintainability_score": 0.85,
                            "documentation_score": 0.8,
                            "test_coverage_score": 0.7,
                            "has_readme": True,
                        },
                        "security_analysis": {
                            "vulnerabilities": [],
                            "security_measures": {
                                "input_validation": {
                                    "present": True,
                                    "matches": [{"file": "src/app.py", "line": 1}],
                                },
                                "authentication": {"present": False, "matches": []},
                                "encryption": {"present": False, "matches": []},
                                "secure_headers": {"present": False, "matches": []},
                                "csrf_protection": {"present": False, "matches": []},
                            },
                            "dependency_security": {"total_vulnerabilities": 0},
                        },
                    },
                }
            ]
        }
    )

    repo_score = result["quality_scores"]["https://github.com/example/repo"]
    assert result["quality_score"] > 0.0
    assert repo_score["overall_score"] > 0.0
    assert repo_score["scores"]["documentation"] >= 0.8
