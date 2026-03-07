.PHONY: bench-roi

bench-roi:
	@if [ -z "$$OPENAI_API_KEY$$ANTHROPIC_API_KEY$$OPENROUTER_API_KEY$$NVIDIA_MINIMAX_API_KEY$$NVIDIA_GLM_API_KEY" ]; then \
		echo "Missing API key: set OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, NVIDIA_MINIMAX_API_KEY, or NVIDIA_GLM_API_KEY"; \
		exit 1; \
	fi
	PYTHONPATH=src python evals/memory/bench_roi.py --output evals/reports/memory_roi_benchmark.json
