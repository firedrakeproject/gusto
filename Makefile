# Adds file annotations to Github Actions (only useful on CI)
GITHUB_ACTIONS_FORMATTING=0
ifeq ($(GITHUB_ACTIONS_FORMATTING), 1)
	FLAKE8_FORMAT=--format '::error file=%(path)s,line=%(row)d,col=%(col)d,title=%(code)s::%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
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
	@echo "    Linting gusto notebooks"
	@python3 -m nbqa flake8 jupyter_notebooks/*.ipynb $(FLAKE8_FORMAT)

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
	@python3 -m pytest examples -v -m "not parallel" $(PYTEST_ARGS)

parallel_example:
	@echo "    Running all parallel examples"
	@python3 -m pytest examples -v -m "parallel" $(PYTEST_ARGS)

notebook_test: clean_cache
	@echo "    Running all Jupyter notebooks"
	@python3 -m pytest --nbval-lax -v jupyter_notebooks $(PYTEST_ARGS)

reset_notebooks:
	@jupyter-nbconvert --clear-output ./jupyter_notebooks/*.ipynb
	@rm -rf ./jupyter_notebooks/results
	@env OMP_NUM_THREADS=1 jupyter-nbconvert \
		--execute \
		--ClearMetadataPreprocessor.enabled=True \
		--allow-errors \
		--to notebook \
		--inplace \
		./jupyter_notebooks/*.ipynb
