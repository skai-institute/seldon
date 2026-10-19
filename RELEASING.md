# Releasing `seldon_core`

This repository is installable as the `seldon_core` distribution. Releases are
plain git tags on `main`; there is no package index upload.

## Tag naming

- Tags are `vX.Y.Z` and must match the `version` in `pyproject.toml`
  (for example `version = "1.2.0"` is tagged `v1.2.0`).
- Versions follow semantic versioning: bump the patch for fixes, the minor for
  backward-compatible additions, the major for breaking changes.

## Who tags

Only the SkAI lead creates tags on `skai-institute/seldon`. Contributors and
the sync tool never push tags.

## Making a release

1. Bump `version` in `pyproject.toml` in its own change
   (`chore: bump seldon_core version to X.Y.Z`) and merge it to `main`.
2. Tag the resulting merge commit on `main` with an annotated tag:

   ```bash
   git tag -a vX.Y.Z -m "seldon_core X.Y.Z" <merge-commit>
   git push origin vX.Y.Z
   ```

Bump first, then tag: a tag whose `pyproject.toml` still carries the previous
version installs a package that reports the wrong version.

## Installing a release

Create the conda environment from `TAE/environment.yaml` first (the PyTorch /
Lightning / CUDA stack comes from conda, not from `pyproject.toml`), then:

```bash
pip install "seldon_core @ git+https://github.com/skai-institute/seldon@vX.Y.Z"
```

Until the first tag exists, install from `main` instead:

```bash
pip install "seldon_core @ git+https://github.com/skai-institute/seldon@main"
```

## Relationship to the private development repository

SELDON is developed in a private repository and mirrored here by
`tools/tfm_sync/` (see its README). Each public tag corresponds to a tag of the
same name in the private repository: the first public tag is `v1.2.0`, matching
the private `v1.2.0`. Commit SHAs differ between the two repositories because
the public history is a filtered rewrite, and the public tree differs in its
README, its BSD-3 LICENSE, the paper assets, and the sync tool itself.

## Merging sync pull requests

Pull requests created by the sync tool must be merged with **Create a merge
commit**. Squash or rebase merging replaces the imported commits, discards the
imported history, and makes the next sync stop with an error.
