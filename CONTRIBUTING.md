# Contributing

Bug reports, feature requests and pull requests are welcome. Please
[open an issue](https://github.com/kalfasyan/plakakia/issues) first for larger changes.

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/kalfasyan/plakakia
cd plakakia
uv sync                      # creates .venv with the package and all dev tools
uv run pre-commit install    # lint and format on every commit
```

## Everyday commands

```bash
uv run pytest                       # tests (+ doctests)
uv run pytest --cov                 # with coverage
uv run ruff check --fix .           # lint
uv run ruff format .                # format
uv run mypy                         # type-check (strict)
uv run mkdocs serve                 # docs at http://127.0.0.1:8000
uv run python benchmarks/bench_pipeline.py --workers 1 4
```

## Guidelines

- Keep `plakakia.tiling` pure NumPy (no I/O); file handling belongs in `io`,
  `annotations` and `pipeline`.
- Add tests for every behaviour change, and type hints and Google-style docstrings for
  every public function.
- Add an entry to `CHANGELOG.md` under *Unreleased*.

## Releasing

1. Bump `version` in `pyproject.toml` (`uv version --bump minor`) and update the changelog.
2. Create a GitHub release tagged `vX.Y.Z`. The `Release` workflow builds the package and
   publishes it to PyPI via trusted publishing.
