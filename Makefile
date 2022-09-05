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

test: unit_test integration_test example notebook_test

clean_cache:
	@echo "    Cleaning caches"
	@firedrake-clean

unit_test: clean_cache
	@echo "    Running all unit-tests"
	@python3 -m pytest unit-tests $(PYTEST_ARGS)

integration_test: clean_cache
	@echo "    Running all integration-tests"
	@python3 -m pytest integration-tests $(PYTEST_ARGS)

example: clean_cache
	@echo "    Running all examples"
	@python3 -m pytest examples $(PYTEST_ARGS)

notebook_test: clean_cache
	@echo "    Running all Jupyter notebooks"
	@python3 -m pytest --nbval-lax -n 4 --dist loadscope jupyter_notebooks $(PYTEST_ARGS)
