from __future__ import annotations

import pytest

from paperbot.utils.analyzer import CodeAnalyzer


@pytest.mark.asyncio
async def test_code_analyzer_returns_stable_contracts(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    (tmp_path / "README.md").write_text(
        "# Sample Repo\n\n## Usage\n\n```bash\npytest\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "fastapi==0.109.0\npytest==8.0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "app.py").write_text(
        '"""App module."""\n'
        "from pydantic import BaseModel\n\n"
        "class InputPayload(BaseModel):\n"
        "    name: str\n\n"
        "def handle(payload):\n"
        '    """Handle payload."""\n'
        "    if payload:\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_app.py").write_text(
        "def test_ok():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    analyzer = CodeAnalyzer({})

    structure = await analyzer.analyze_structure(tmp_path)
    security = await analyzer.analyze_security(tmp_path)
    quality = await analyzer.analyze_quality(tmp_path)
    dependencies = await analyzer.analyze_dependencies(tmp_path)

    assert structure["primary_language"] == "Python"
    assert "src/app.py" in structure["files"]["file_paths"]
    assert structure["documentation"]["has_readme"] is True
    assert dependencies["direct_dependencies"]["python"][0]["name"] == "fastapi"

    assert security["security_measures"]["input_validation"]["present"] is True
    assert security["dependency_security"]["scanner"] == "safety"

    assert quality["documentation_metrics"]["has_readme"] is True
    assert quality["overall_score"] > 0

