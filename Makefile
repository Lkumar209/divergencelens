.PHONY: install lint type test smoke bench serve reproduce clean

install:
	uv sync

lint:
	uv run ruff check src/ bench/ tests/
	uv run ruff format --check src/ bench/ tests/

fmt:
	uv run ruff format src/ bench/ tests/
	uv run ruff check --fix src/ bench/ tests/

type:
	uv run mypy src/divergencelens/

test:
	uv run pytest tests/unit/ -q --tb=short

smoke:
	uv run pytest tests/smoke/ -q --tb=short -v

bench:
	uv run divergencelens bench --split test --seeds 3 --output-dir results/

serve:
	uv run divergencelens serve --host 0.0.0.0 --port 8000

reproduce:
	@echo "Regenerating DivergenceBench and computing all metrics..."
	uv run python -m bench.metrics.compute
	@echo "Done. See results/RESULTS.md"

clean:
	rm -rf .cache/ results/ dist/ build/ __pycache__ .mypy_cache .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

ci: lint type test smoke
