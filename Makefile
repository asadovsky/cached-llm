SHELL := /bin/bash -euo pipefail

.DELETE_ON_ERROR:

.PHONY: test
test:
	python -m unittest discover . '*_test.py'

########################################
# Format and lint

.PHONY: fmt
fmt:
	@uvx --from scm-kit scm-format

.PHONY: lint
lint:
	@uvx --from scm-kit scm-lint

.PHONY: lint-all
lint-all:
	@uvx --from scm-kit scm-lint -a
