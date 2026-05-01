# Release Checklist

Use this checklist before publishing a Matheel release.

## Version

- Confirm `pyproject.toml` has the intended version.
- Confirm the release tag uses the same version with a leading `v`, for example `v0.3.4`.
- Confirm release notes describe the user-visible changes.

## Tests

- Run the default offline-friendly test suite:

```bash
pytest
```

- Run real-model integration tests when optional backends, cached model weights, or network access are available:

```bash
pytest -m integration
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

## GitHub Release

- Confirm the tag exists on GitHub.
- Publish the GitHub Release for the same tag.
- Mark the release as latest when it is the current stable release.
- Confirm GitHub Releases, repository tags, and `pyproject.toml` agree:

```bash
gh release list --limit 5
git tag --sort=-version:refname | head
```

## Post-Release

- Install the released package in a clean environment.
- Run a small CLI check against `sample_pairs.zip`.
- Open an issue for any follow-up that should not block the release.
