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

# Disable threading as this can slow down tests
export OMP_NUM_THREADS=1

# Run parallel tests:
#  - The first argument specifies the number of processors
#  - The second argument specifies the directory to search for tests
define run_parallel_tests
	@mpiexec -n $(1) python3 -m pytest $(2) -m parallel[match] $(PYTEST_ARGS)
endef

# Run all unit tests assuming they are all serial
define run_unit_test
	@echo "    Running all unit-tests"
	@python3 -m pytest unit-tests $(PYTEST_ARGS)
endef

# Run all integration tests
# There are currently no integration tests using more than 2 MPI ranks. Add a call to run_parallel_tests
# below if such a test is added in future
define run_integration_test
	@echo "    Running all integration-tests"
	@python3 -m pytest integration-tests -m parallel[match] $(PYTEST_ARGS)
	$(call run_parallel_tests,2,integration-tests)
endef

# Run serial examples i.e 'not parallel' or nprocs=1
define run_example
	@echo "    Running serial examples"
	@python3 -m pytest examples -m parallel[match] $(PYTEST_ARGS)
endef

# Run parallel examples using 2, 3 or 4 processors
# If a test is added with more than 4 processors then add the relevant call to run_parallel_tests below
define run_parallel_example
	@echo "    Running all parallel examples"
	$(call run_parallel_tests,2,examples)
	$(call run_parallel_tests,3,examples)
	$(call run_parallel_tests,4,examples)
endef

test:
	@echo "    Running all tests"
	$(run_unit_test)
	$(run_integration_test)
	$(run_example)
	$(run_parallel_example)

unit_test:
	$(run_unit_test)

integration_test:
	$(run_integration_test)

example:
	$(run_example)

parallel_example:
	$(run_parallel_example)
