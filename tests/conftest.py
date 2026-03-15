# tests/conftest.py
"""
Pytest configuration and fixtures.
Adds the project `src` directory to `sys.path` so `paperbot` imports resolve in tests.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
# Add src to path so `import paperbot` works
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# Also add project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Exclude stale root-level tests pending dedicated cleanup PRs.
collect_ignore = [
    str(Path(__file__).parent / "test_code_analysis_fallback.py"),
    str(Path(__file__).parent / "test_conference_agent_stats.py"),
    str(Path(__file__).parent / "test_influence_recency.py"),
    str(Path(__file__).parent / "test_literature_grounding.py"),
    str(Path(__file__).parent / "test_render_report_latest.py"),
]
