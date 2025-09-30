# vcsencode (scientific scaffold)

Modular, reversible Vessel Coordinate System (VCS) pipeline.

## Development Quickstart

Environment is assumed active. Typical commands:

```bash
# Install package in editable mode (for CLI entry point)
pip install -e .

# Install pre-commit hooks (format/lint on commit)
pre-commit install

# Run lint and tests
make lint
make test

# CLI (stubs until later prompts)
vcsencode --help
```

Artifacts (later):
- `model.npy`, `meta.json` — encoding and metadata
- `fig3.png` / `fig3.svg` — Figure (from encoding only)
- `reconstructed.stl` — decoded mesh