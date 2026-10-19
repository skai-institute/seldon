# tfm_sync: mirror the private development repository

SELDON is developed in a private repository (TFM). This tool brings new commits
from TFM's `main` branch into this public repository as a pull request, with
the history rewritten so that only allowed paths are ever published.

What the tool guarantees:

- Only paths present at TFM's tip (plus paths already published) appear in
  imported commits. TFM's `README.md`, `docs/plans/` and `docs/brainstorms/`
  are never imported (see `NEVER_IMPORTED` in `sync.py`).
- Authors, committers, dates and messages are unchanged.
- Commits already public are never rewritten; the state file
  `tools/tfm_sync/state.json` maps each TFM commit to its public commit.
- Public-owned files (`README.md`, `LICENSE`, `SELDONI.pdf`, `SELDON.png`,
  `RELEASING.md`, `tools/tfm_sync/`, `tests/test_tfm_sync.py`) always come from
  the public side. On every other path, TFM wins.
- Rewriting happens in a throwaway repository, so filtered-out private objects
  never enter your public checkout.
- Before the sync branch is created, everything it adds (file contents, path
  names, and complete commit objects including author and committer headers) is
  scanned for private clone URLs, private repository URLs, SSH key fingerprints
  and SSH directory paths. Any hit stops the run.

## Prerequisites

- Python 3.12 and git (no other dependencies).
- A TFM checkout with an up-to-date `origin/main` (`git fetch origin`). The tool
  reads that ref only, never your worktree or `HEAD`, so unpushed work is never
  published.
- This public checkout with `upstream` pointing at `skai-institute/seldon`,
  freshly fetched (`git fetch upstream`), and a `bot` remote pointing at the
  machine-account fork.
- Only one sync pull request open at a time. Close or merge the previous one
  before starting another.

## Running a sync

From the public repository root:

1. Create a base branch from public main:

   ```bash
   git fetch upstream
   git switch -c <BASE-BRANCH> upstream/main
   ```

   Commit any public-owned edits (for example README reconciliation) here.

2. Run the sync:

   ```bash
   python -m tools.tfm_sync sync --source <TFM-CHECKOUT> \
     --base <BASE-BRANCH> --branch <SYNC-BRANCH>
   ```

   `--source-ref` (default `origin/main`) and `--public-ref` (default
   `upstream/main`) can be overridden.

3. Read the report:

   - **commits written / dropped**: how many TFM commits were imported.
   - **imported paths edited or deleted on public main**: someone changed or
     deleted an imported file directly on public main; the TFM version overwrites
     or restores it. Move the change into TFM if it should survive.
   - **paths the sync merge deletes**: files on public main that are neither
     public-owned nor in TFM. Move them into TFM, or add them to the
     public-owned list in `sync.py`, before pushing.
   - **TFM README.md changes**: reconcile by hand on the base branch (rerun the
     sync afterwards) or as a follow-up commit on the sync branch.

4. Scan the exact head you are about to push. This must be the last command
   before every push, and must be rerun after any manual commit or amend:

   ```bash
   python -m tools.tfm_sync scan --branch <SYNC-BRANCH>
   ```

5. Verify identity, then push only the sync branch to the fork:

   ```bash
   gh api user --jq .login           # must print skoranda-agent
   git remote get-url --push --all bot   # must be github.com/skoranda-agent/seldon
   git push bot <SYNC-BRANCH>
   ```

6. Open the pull request against `skai-institute/seldon` `main` and state in it
   that it must be merged with **Create a merge commit**.

## Exit codes and stops

| Exit | Meaning |
|---|---|
| 0 | Sync branch created, or nothing to sync (no branch is created). |
| 2 | A guard or the scan stopped the run. Nothing was written to this checkout. |

Stops and what to do:

- **Leak scan hit**: the message names the pattern, where it was found and the
  object. Never push. If the hit is on the base branch, fix it there and rerun.
  If it is in TFM content, fix it in TFM and rerun; but when the string sits in
  an *old* TFM commit at an allowed path, rerunning hits it again because TFM
  history is fixed. In that case keep the path out of public history by adding
  it to `NEVER_IMPORTED` in `sync.py` (reviewed in the same PR), or stop and
  ask the TFM maintainers.
- **TFM main was force-pushed**: the recorded TFM tip is no longer in TFM's
  history. Published commits cannot change; ask the TFM maintainers to restore
  the old history (push the old tip back to TFM main) and rerun.
- **Previous sync PR was merged by squash or rebase**: the last synced commit is
  not in public main's history. See the recovery steps below.
- **Branch already exists**: pick a new `--branch` name.

### Recovering from a squash or rebase merge

If a sync PR was squash- or rebase-merged, public main contains the right files
but not the rewritten commits the state file points at. The supported recovery
is to revert that squash commit on public main (through a normal PR), then rerun
the sync so the same rewritten commits arrive through a merge commit. Do not
edit `state.json` by hand: its commit map must point only at commits that public
main actually contains, or later syncs fail.
