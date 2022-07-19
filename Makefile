lint:
	@echo "    Linting gusto codebase"
	@python3 -m flake8 gusto
	@echo "    Linting gusto examples"
	@python3 -m flake8 examples
	@echo "    Linting gusto unit-tests"
	@python3 -m flake8 unit-tests
	@echo "    Linting gusto integration-tests"
	@python3 -m flake8 integration-tests
	@echo "    Linting gusto plotting scripts"
	@python3 -m flake8 plotting

test:
	@echo "    Running all tests"
	@python3 -m pytest unit-tests integration-tests $(PYTEST_ARGS)

unit_test:
	@echo "    Running all unit-tests"
	@python3 -m pytest unit-tests $(PYTEST_ARGS)

integration_test:
	@echo "    Running all integration-tests"
	@python3 -m pytest integration-tests $(PYTEST_ARGS)
