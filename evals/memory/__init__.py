"""
Memory evaluation entrypoints for scope, safety, and manual ROI benchmarking.

Test files:
- test_deletion_compliance.py: Verify deleted items never retrieved
- test_retrieval_hit_rate.py: Verify relevant memories are found
- test_scope_isolation.py: Verify zero cross-user and cross-scope leakage
- test_injection_robustness.py: Verify offline prompt-injection pattern detection
"""
