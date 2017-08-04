lint:
	@echo "    Linting gusto codebase"
	@python3 -m flake8 gusto
	@echo "    Linting gusto examples"
	@python3 -m flake8 examples
	@echo "    Linting gusto tests"
	@python3 -m flake8 tests

test:
	@echo "    Running all tests"
	@python3 -m pytest tests $(PYTEST_ARGS)
