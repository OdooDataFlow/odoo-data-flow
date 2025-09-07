# Odoo Data Flow - CRUSH.md

## Build/Lint/Test Commands
- Setup: `nox -s setup` (if available) or `uv sync`
- Lint: `nox -s pre-commit` or `ruff check .` and `ruff format .`
- Type check: `nox -s mypy` or `mypy src tests docs/conf.py`
- Run all tests: `nox -s tests` or `pytest tests/`
- Run single test: `pytest tests/test_file.py::test_function`
- Coverage: `nox -s coverage`
- Docs: `nox -s docs` (live) or `nox -s docs-build` (build)

## Code Style Guidelines
- Formatting: ruff (88 char line limit)
- Imports: isort style, grouped by standard library, third-party, local
- Types: Use lowercase built-ins (list, dict), fully type-hinted code
- Naming: snake_case for variables/functions, PascalCase for classes
- Docstrings: Google style (Args:, Returns:) with one-line summary
- Error handling: Use exceptions from src/odoo_data_flow/lib/internal/exceptions.py
- Tests: Pytest with descriptive names, follow existing patterns in tests/

## Project Structure
- src/odoo_data_flow/: Main source code
- tests/: Unit and integration tests
- docs/: Documentation source
- conf/: Configuration files
- testdata/: Sample data files
- .nox/: Virtual environments
- Key files: noxfile.py, pyproject.toml, .pre-commit-config.yaml
