# Contributing

Thanks for considering a contribution to `torch-floating-point`.

## Development setup

```bash
git clone https://github.com/SamirMoustafa/torch-floating-point.git
cd torch-floating-point
make env
# or: pip install -e ".[dev]" && pip install pytest parameterized numpy
```

The C++/CUDA extension builds at install time. A CPU extension is enough for the default test suite. CUDA kernels compile when the machine has a GPU toolchain; GitHub Actions CI installs CPU PyTorch only and does **not** validate CUDA.

## Workflow

1. Open an issue for bugs or substantial changes before a large pull request.
2. Fork the repository and create a branch from `main`.
3. Add or update tests under `test/` for behavioral changes.
4. Run the local checks:

   ```bash
   make lint
   make test
   ```

   Format with `make format` if Ruff reports style issues.
5. Open a pull request. Describe the change, the tests you ran, and whether CUDA was involved.

## What belongs in a PR

- Codec or rounding changes should come with goldens or unit tests. NVIDIA decode tables live in `test/nvidia_codec_goldens.py` (regenerated from CUDA headers; do not edit by hand).
- Named hardware formats belong in documentation and tests, not as new public aliases, unless a maintainer agrees otherwise.
- Do not add benchmark plots or accuracy tables that the project cannot reproduce in CI.

## Reporting issues

Use [GitHub Issues](https://github.com/SamirMoustafa/torch-floating-point/issues). Include the package version (`python -c "import version; print(version.__version__)"`), PyTorch version, OS, and whether you used CPU or CUDA.

Questions that are not bugs fit [GitHub Discussions](https://github.com/SamirMoustafa/torch-floating-point/discussions).

## License

Contributions are accepted under the MIT License in `LICENSE`.
