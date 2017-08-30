lint:
	@echo "    Linting gusto codebase"
	@python3 -m flake8 gusto
	@echo "    Linting gusto examples"
	@python3 -m flake8 examples
	@echo "    Linting gusto tests"
	@python3 -m flake8 tests
	@echo "    Linting gusto plotting scripts"
	@python3 -m flake8 plotting

test:
	@echo "    Running all tests"
	@python3 -m pytest tests $(PYTEST_ARGS)
