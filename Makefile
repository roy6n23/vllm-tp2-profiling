PY ?= .venv/bin/python
.PHONY: test dry-run check contract predict
test:
	$(PY) -m pytest -q
dry-run:
	rm -rf results/dryrun && $(PY) -m tpprof run --tier all --dry-run --results-dir results/dryrun && $(PY) -m tpprof analyze --results-dir results/dryrun && $(PY) -m tpprof report --results-dir results/dryrun
check: test dry-run
contract:
	docker build -f docker/contract.Dockerfile -t tpprof-contract . && docker run --rm tpprof-contract
predict:
	$(PY) -m tpprof predict --out predictions.md
