SHELL := /bin/bash -euo pipefail

.DELETE_ON_ERROR:

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
