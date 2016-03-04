lint:
	@echo "    Linting dcore codebase"
	@flake8 dcore
	@echo "    Linting dcore examples"
	@flake8 examples
