lint:
	@echo "    Linting gusto codebase"
	@flake8 gusto
	@echo "    Linting gusto examples"
	@flake8 examples
	@echo "    Linting gusto tests"
	@flake8 tests
	@echo "    Linting gusto plotting scripts"
	@flake8 tests

test:
	@echo "    Running all tests"
	@py.test tests $(PYTEST_ARGS)
