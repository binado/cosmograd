# cosmograd

Differentiable cosmology in JAX behind a small protocol contract.

## Setup

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Just](https://just.systems/man/en/packages.html), then run:

```console
just setup
uv run prek install
```

The underlying command is `uv sync --all-groups` if Just is unavailable.



## Development

```console
just fmt        # uv run ruff check --fix . && uv run ruff format .
just lint       # uv run ruff check .
just typecheck  # uv run ty check
just test       # uv run pytest
just hooks      # uv run prek run --all-files
just check      # full validation, tests, and package build
```

## Update from the template

The template runs `uv lock`, so updates must explicitly trust it:

```console
uvx copier@9.18.2 update --trust
just check
```

