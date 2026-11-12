# Repository Guidelines

## Dev workflows

- `just setup`: install development and test dependencies with `uv sync --all-groups`.
- `uv run prek install`: enable local commit hooks.
- `just fmt`: apply Ruff lint fixes and formatting.
- `just lint` and `just typecheck`: run Ruff and ty checks.
- `just test`: run pytest; use `just test tests/test_background.py -k gradient` for focused checks.
- `just build`: build source and wheel distributions with `uv build`.
- `just check`: run lint, formatting validation, type checks, tests, build, and lockfile validation before submitting changes.

Recipes are defined in `justfile`; use their underlying `uv` commands if Just
is unavailable.

## Code styleguide

- Add type annotations and follow existing NumPy-style docstrings
- Prefer common jax design patterns
- Prefer documenting conventions in docstrings than enforcing array argument validation

## Testing styleguide

- Use pytest with `test_*.py` files and `test_*` functions.
- Avoid tests that repeat the function implementation
- Test array comparisons with `np.testing.assert_allclose`
- Avoid tests that require fine-tuned tolerances
- Avoid creating mock objects and monkeypatching
- Tests enable JAX float64 in `tests/conftest.py`; avoid changing global precision
in library code. CI tests Python 3.12–3.14

## Commit & Pull Request Guidelines

- Follow the existing Conventional Commit pattern, such as `feat: add background
implementation` or `chore: update tooling`.
