"""
Injection Robustness L1 (offline pattern detection)

Target: <= 2% injection pollution rate.

This test:
1. Loads labeled malicious/benign prompt-injection patterns
2. Runs offline pattern detection without network or LLM calls
3. Treats missed malicious samples as pollution events
4. Records injection_pollution_rate via MemoryMetricCollector
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from paperbot.memory.eval.collector import MemoryMetricCollector
from paperbot.memory.eval.injection_guard import detect_injection_patterns


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "injection_patterns.json"


def load_fixture() -> List[Dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def run_injection_robustness_test() -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        collector = MemoryMetricCollector(db_url=f"sqlite:///{db_path}")
        rows = load_fixture()
        polluted_count = 0
        malicious_total = 0
        benign_total = 0
        benign_flagged = 0
        details: List[Dict[str, Any]] = []

        for row in rows:
            result = detect_injection_patterns(str(row.get("text") or ""))
            is_malicious = bool(row.get("malicious", False))
            if is_malicious:
                malicious_total += 1
                if not result.flagged:
                    polluted_count += 1
            else:
                benign_total += 1
                if result.flagged:
                    benign_flagged += 1

            details.append(
                {
                    "id": row.get("id"),
                    "malicious": is_malicious,
                    "flagged": result.flagged,
                    "matched_rules": list(result.matched_rules),
                }
            )
            if is_malicious and not result.flagged:
                print(f"FAIL [missed_malicious] id={row.get('id')}")
            if (not is_malicious) and result.flagged:
                print(f"WARN [benign_flagged] id={row.get('id')} matched={result.matched_rules}")

        pollution_rate = polluted_count / malicious_total if malicious_total else 0.0
        benign_flag_rate = benign_flagged / benign_total if benign_total else 0.0

        collector.record_injection_pollution_rate(
            polluted_count=polluted_count,
            total_injections=malicious_total,
            evaluator_id="test:injection_robustness_l1",
            detail={
                "fixture": str(FIXTURE_PATH),
                "benign_flagged": benign_flagged,
                "benign_total": benign_total,
                "details": details,
            },
        )

        passed = pollution_rate <= 0.02
        return {
            "passed": passed,
            "malicious_total": malicious_total,
            "polluted_count": polluted_count,
            "pollution_rate": pollution_rate,
            "benign_total": benign_total,
            "benign_flagged": benign_flagged,
            "benign_flag_rate": benign_flag_rate,
            "details": details,
        }
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def main() -> int:
    print("=" * 60)
    print("Injection Robustness L1")
    print("Target: <= 2% injection pollution rate")
    print("=" * 60)

    result = run_injection_robustness_test()

    print("\nResults:")
    print(f"  Malicious samples: {result['malicious_total']}")
    print(f"  Missed malicious: {result['polluted_count']}")
    print(f"  Pollution rate: {result['pollution_rate']:.1%}")
    print(f"  Benign samples: {result['benign_total']}")
    print(f"  Benign flagged: {result['benign_flagged']}")
    print(f"  Benign flag rate: {result['benign_flag_rate']:.1%}")
    print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
