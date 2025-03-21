# Adds file annotations to Github Actions (only useful on CI)
GITHUB_ACTIONS_FORMATTING=0
ifeq ($(GITHUB_ACTIONS_FORMATTING), 1)
	FLAKE8_FORMAT=--format='::error file=%(path)s,line=%(row)d,col=%(col)d,title=%(code)s::%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
else
	FLAKE8_FORMAT=
endif

lint:
	@echo "    Linting gusto codebase"
	@python3 -m flake8 $(FLAKE8_FORMAT) gusto
	@echo "    Linting gusto examples"
	@python3 -m flake8 $(FLAKE8_FORMAT) examples
	@echo "    Linting gusto unit-tests"
	@python3 -m flake8 $(FLAKE8_FORMAT) unit-tests
	@echo "    Linting gusto integration-tests"
	@python3 -m flake8 $(FLAKE8_FORMAT) integration-tests
	@echo "    Linting gusto plotting scripts"
	@python3 -m flake8 $(FLAKE8_FORMAT) plotting

test:
	@echo "    Running all tests"
	@python3 -m pytest -q -s unit-tests integration-tests examples $(PYTEST_ARGS) --show-capture=no

unit_test:
	@echo "    Running all unit-tests"
	@python3 -m pytest -q -s unit-tests $(PYTEST_ARGS) --show-capture=no

integration_test:
	@echo "    Running all integration-tests"
	@python3 -m pytest -q -s integration-tests $(PYTEST_ARGS) --show-capture=no

example:
	@echo "    Running all examples"
	@python3 -m pytest -q -s examples -v -m "not parallel" $(PYTEST_ARGS) --show-capture=no

parallel_example:
	@echo "    Running all parallel examples"
	@python3 -m pytest -q -s examples -v -m "parallel" $(PYTEST_ARGS) --show-capture=no
