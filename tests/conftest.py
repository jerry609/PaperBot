# tests/conftest.py
"""
Pytest configuration and fixtures.
Adds src/paperbot to sys.path so imports work correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
# Add src to path so `import paperbot` works
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# Keep legacy behavior: add src/paperbot so `import repro` style tests keep working
legacy_src_path = src_root / "paperbot"
if str(legacy_src_path) not in sys.path:
    sys.path.insert(0, str(legacy_src_path))

# Also add project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Exclude legacy test files that use old import paths (agents.xxx, repro.xxx, etc.)
collect_ignore = [
    str(Path(__file__).parent / "test_code_analysis_fallback.py"),
    str(Path(__file__).parent / "test_conference_agent_stats.py"),
    str(Path(__file__).parent / "test_generation_agent.py"),
    str(Path(__file__).parent / "test_influence_recency.py"),
    str(Path(__file__).parent / "test_literature_grounding.py"),
    str(Path(__file__).parent / "test_render_report_latest.py"),
    str(Path(__file__).parent / "test_repro_planning.py"),
]
