# tests/conftest.py
"""
Pytest configuration and fixtures.
Adds src/paperbot to sys.path so imports work correctly.
"""

import sys
from pathlib import Path

# Add src/paperbot to path so 'repro' imports work
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "paperbot"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
