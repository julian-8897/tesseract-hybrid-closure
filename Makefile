.PHONY: compile lint format test gradient-smoke rollout-smoke container-smoke build-tesseracts demo demo-images notebook notebook-browser notebook-public results spectra spectra-figure decomposition-figure served-training served-training-figure hooks verify

verify: compile lint test gradient-smoke rollout-smoke
	@echo "verify: all checks passed"

compile:
	uv run python -m compileall -q src tests tesseracts notebooks

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run python -m pytest tests -q -m "not gradient_smoke and not rollout_smoke and not container"

gradient-smoke:
	uv run python -m pytest tests/test_gradient_smoke.py -q -m gradient_smoke
	uv run hybrid-closure gradient-smoke --output artifacts/gradient-smoke.json

rollout-smoke:
	uv run python -m pytest tests/test_rollout_smoke.py -q -m rollout_smoke

demo:
	uv run hybrid-closure demo

demo-images:
	uv run hybrid-closure demo --images

notebook:
	uv run marimo edit notebooks/visual_walkthrough.py

notebook-browser:
	uv run marimo edit notebooks/browser_walkthrough.py

notebook-public:
	uv run python scripts/generate_notebook_demo_data.py

container-smoke:
	uv run python -m pytest tests/test_container_composition.py -q -m container

build-tesseracts:
	./buildall.sh

results:
	uv run python scripts/generate_submission_assets.py

# Validation-split spectral diagnostic. Rolls every method over the whole
# validation split, so it takes appreciably longer than the smokes.
spectra:
	uv run hybrid-closure spectra \
	  --checkpoint runs/w2-calibrated-a20-dt002-100x3/stage-unroll-30-updates-700.pkl \
	  --apriori-checkpoint runs/final-submission/apriori-700/checkpoint.pkl \
	  --report-steps 30 120 500 \
	  --include-smagorinsky \
	  --output docs/results/spectra-validation.json

spectra-figure:
	uv run python scripts/generate_spectra_figure.py

decomposition-figure:
	uv run python scripts/generate_decomposition_figure.py

# A-posteriori training whose every gradient crosses the served images. Needs
# Docker and the built Tesseracts; takes roughly five minutes for 500 updates.
# Each invocation writes to a fresh timestamped directory under
# runs/served-training, so reruns never overwrite earlier evidence (preflight
# refuses an existing report or checkpoint before any DNS generation).
ifndef SERVED_RUN_DIR
SERVED_RUN_DIR := runs/served-training/$(shell date +%Y%m%d-%H%M%S)
endif

served-training:
	@uv run hybrid-closure train-served --updates 500 \
	  --reference-checkpoint runs/w2-calibrated-a20-dt002-100x3/stage-unroll-30-updates-700.pkl \
	  --output "$(SERVED_RUN_DIR)/served-training.json" \
	  --checkpoint-output "$(SERVED_RUN_DIR)/served-training-params.pkl" \
	  && echo "served-training evidence written to $(SERVED_RUN_DIR)/"

served-training-figure:
	uv run python scripts/generate_served_training_figure.py

hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit gate installed (core.hooksPath = .githooks)"

format:
	uv run ruff check --fix .
	uv run ruff format .
