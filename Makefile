lint:
	@echo "    Linting dcore codebase"
	@flake8 dcore
	@echo "    Linting dcore examples"
	@flake8 examples
	@echo "    Linting dcore tests"
	@flake8 tests

test:
	@echo "    Running all tests"
	@py.test tests $(PYTEST_ARGS)
