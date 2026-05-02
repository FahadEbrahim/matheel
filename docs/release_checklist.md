# Release Checklist

Use this checklist before publishing a Matheel release.

## Release Flow

- Make release preparation changes on an issue branch.
- Open a pull request, wait for GitHub Actions to pass, and merge to `main`.
- Publish from a GitHub Release, not from a tag push or a manual publish run.
- Keep PyPI publishing tokenless through Trusted Publishing. Do not add PyPI API tokens or repository secrets for publishing.

The publish workflow is intentionally triggered by `release.published`. A tag push alone should not publish to PyPI because release notes and the "latest" setting should be reviewed first. A manual `workflow_dispatch` publish is avoided because it is easier to run from the wrong ref. If a publish run needs to be retried, rerun the failed GitHub Actions job that was created by the GitHub Release event.

## Version

- Confirm `pyproject.toml` has the intended version.
- Confirm the release tag uses the same version with a leading `v`, for example `v0.3.6`.
- Sync the Gradio app version references after changing the package version:

```bash
python scripts/sync_gradio_app_version.py
```

- Confirm the Gradio app README and requirements pin match `pyproject.toml`:

```bash
python scripts/sync_gradio_app_version.py --check
```

- Confirm the GitHub Release notes describe the user-visible changes.
- Do not add release notes under `docs/releases`; keep release notes on GitHub Releases.
- Confirm the README PyPI badges are dynamic and use the standard Shields cache:
  - PyPI version: `https://img.shields.io/pypi/v/matheel.svg`
  - Python versions: `https://img.shields.io/pypi/pyversions/matheel.svg`

## Tests

- Run the default offline-friendly test suite:

```bash
python -m pytest
```

- Run real-model integration tests when optional backends, cached model weights, or network access are available:

```bash
python -m pytest -m integration
```

- Run Ruff:

```bash
python -m ruff check .
```

## Package

- Build the distribution artifacts:

```bash
python -m build
```

- Check the distribution metadata:

```bash
python -m twine check dist/*
```

- Confirm package metadata includes Python classifiers for `3.10`, `3.11`, `3.12`, and `3.13`.

## Documentation

- Confirm the Pages workflow has passed on the latest `main` commit.
- Confirm the published docs open at <https://fahadebrahim.github.io/matheel/>.
- Keep release notes on GitHub Releases instead of publishing them under `docs/releases`.

## GitHub Release and PyPI

- Create and push the release tag from the merged `main` commit.
- Publish the GitHub Release for the same tag. The publish workflow checks that the release tag matches `pyproject.toml`, builds the package, checks the metadata, and uploads to PyPI through Trusted Publishing.
- The Hugging Face Space sync workflow runs after the publish workflow succeeds. It also runs after merged changes to `gradio_app/` on `main`.
- Confirm the repository variable `HF_SPACE_REPO` points to the target Space if the default Space changes.
- Confirm the `HF_TOKEN` secret is configured with write access to the target Space.
- Mark the release as latest when it is the current stable release.
- Confirm PyPI, GitHub Releases, repository tags, and `pyproject.toml` agree:

```bash
python -m pip index versions matheel
gh release list --limit 5
git tag --sort=-version:refname | head
```

- Confirm the GitHub Actions publish run succeeded for the release event.
- Confirm the GitHub Actions Space sync run succeeded for the release event.
- Confirm PyPI shows the new version, the `Requires-Python` range, and Python version classifiers.
- Confirm the README badges show the released PyPI version and supported Python versions after the standard Shields cache refreshes.

## Post-Release

- Install the released package in a clean environment.
- Run a small CLI check against `sample_pairs.zip`.
- Open an issue for any follow-up that should not block the release.
