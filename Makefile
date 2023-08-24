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
	@python3 -m pytest unit-tests integration-tests examples $(PYTEST_ARGS)

unit_test:
	@echo "    Running all unit-tests"
	@python3 -m pytest unit-tests $(PYTEST_ARGS)

integration_test:
	@echo "    Running all integration-tests"
	@python3 -m pytest integration-tests $(PYTEST_ARGS)

example:
	@echo "    Running all examples"
	@python3 -m pytest examples $(PYTEST_ARGS)
