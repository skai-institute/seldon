"""Tests for the TFM -> public sync tool (``tools/tfm_sync``).

Every test builds small throwaway git repositories under ``tmp_path``: a fake
private development repository and a fake public repository. Git identity is
passed explicitly on every call because CI runners have no ``user.name``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.tfm_sync import sync

IDENTITY = {
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.org",
    "GIT_COMMITTER_NAME": "Test Committer",
    "GIT_COMMITTER_EMAIL": "committer@example.org",
}

# Leak-shaped fixture strings, built from parts so this file never matches the
# scanner's own patterns.
FAKE_PRIVATE_CLONE = "git" + "@github.com:" + "skai-institute/" + "seldon-dsb.git"
FAKE_FINGERPRINT = "SHA256" + ":" + "A" * 43


def git(repo: Path, *args: str, date: str = "1700000000 +0000") -> str:
    """Run git in ``repo`` with a fixed identity and date."""
    env = dict(os.environ, **IDENTITY)
    env["GIT_AUTHOR_DATE"] = date
    env["GIT_COMMITTER_DATE"] = date
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def commit(
    repo: Path,
    message: str,
    files: dict[str, str | bytes | None],
    date: str = "1700000000 +0000",
) -> str:
    """Write, delete (``None``) and commit files; return the new commit SHA."""
    for name, content in files.items():
        path = repo / name
        if content is None:
            git(repo, "rm", "-q", name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        git(repo, "add", "-f", name)
    git(repo, "commit", "-q", "--allow-empty", "-m", message, date=date)
    return git(repo, "rev-parse", "HEAD")


def init_repo(path: Path) -> Path:
    """Create an empty repository on branch ``main``."""
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    return path


@pytest.fixture
def private(tmp_path: Path) -> Path:
    """Fake private repo: a deleted checkpoint, a README with a leak, MIT LICENSE."""
    repo = init_repo(tmp_path / "private")
    commit(
        repo,
        "Initial commit",
        {"pkg/core.py": "x = 1\n", "LICENSE": "MIT\n", "README.md": "# Private\n"},
        date="1700000001 +0000",
    )
    commit(
        repo, "Add checkpoint", {"logs/last.ckpt": b"\x00weights"}, "1700000002 +0000"
    )
    commit(
        repo,
        "Document private access",
        {"README.md": f"# Private\nclone {FAKE_PRIVATE_CLONE}\n{FAKE_FINGERPRINT}\n"},
        "1700000003 +0000",
    )
    commit(
        repo,
        "Remove checkpoint and grow core",
        {"logs/last.ckpt": None, "pkg/core.py": "x = 2\n"},
        "1700000004 +0000",
    )
    # Expose the tip as origin/main, the ref the tool reads by default.
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


@pytest.fixture
def public(tmp_path: Path) -> Path:
    """Fake public repo with paper assets, BSD LICENSE and public-only code."""
    repo = init_repo(tmp_path / "public")
    commit(
        repo,
        "Initial public snapshot",
        {
            "README.md": "# SELDON\n",
            "LICENSE": "BSD-3\n",
            "SELDON.png": b"\x89PNG fake",
            "pkg/core.py": "x = 0\n",
            "plot_realtime.py": "print('old')\n",
        },
        "1690000000 +0000",
    )
    git(repo, "update-ref", "refs/remotes/upstream/main", "HEAD")
    git(repo, "switch", "-q", "-c", "base")
    commit(
        repo, "Add sync tool", {"tools/tfm_sync/__init__.py": ""}, "1690000001 +0000"
    )
    return repo


def run(
    private: Path, public: Path, branch: str = "sync-1", base: str = "base"
) -> sync.SyncResult:
    """Run one sync with the test defaults."""
    return sync.run_sync(
        source=private,
        public=public,
        base=base,
        branch=branch,
        source_ref="origin/main",
        public_ref="upstream/main",
    )


def paths_ever_touched(repo: Path, rev_range: str) -> set[str]:
    """Every path any commit in ``rev_range`` added, changed or deleted."""
    out = git(repo, "log", "--format=", "--name-only", rev_range)
    return {line for line in out.splitlines() if line}


def merge_sync_branch(public: Path, branch: str) -> None:
    """Simulate the lead merging the sync PR into public main with a merge commit."""
    git(public, "switch", "-q", "main")
    git(public, "merge", "-q", "--no-ff", "-m", "Merge sync PR", branch)
    git(public, "update-ref", "refs/remotes/upstream/main", "main")
    git(public, "switch", "-q", "-c", f"base-after-{branch}")


# --- U1: allowlist, rewrite and state -------------------------------------


def test_first_sync_never_publishes_deleted_or_readme_paths(
    private: Path, public: Path
) -> None:
    result = run(private, public)
    assert result.status == "synced"
    touched = paths_ever_touched(public, f"upstream/main..{result.branch}^2")
    assert "logs/last.ckpt" not in touched
    assert "README.md" not in touched
    assert touched == {"pkg/core.py", "LICENSE"}


def test_rewritten_commits_keep_identity_dates_and_messages(
    private: Path, public: Path
) -> None:
    result = run(private, public)
    src = git(private, "log", "--format=%an|%ae|%ad|%cn|%ce|%cd|%B", "--date=raw", "-1")
    out = git(
        public,
        "log",
        "--format=%an|%ae|%ad|%cn|%ce|%cd|%B",
        "--date=raw",
        "-1",
        f"{result.branch}^2",
    )
    assert out == src


def test_commit_touching_only_disallowed_paths_is_dropped(
    private: Path, public: Path
) -> None:
    result = run(private, public)
    subjects = git(public, "log", "--format=%s", f"upstream/main..{result.branch}^2")
    assert "Add checkpoint" not in subjects.splitlines()
    assert "Document private access" not in subjects.splitlines()
    assert result.written == 2
    assert result.dropped == 2


def test_rewrite_is_deterministic(private: Path, public: Path) -> None:
    first = run(private, public, branch="sync-a")
    second = run(private, public, branch="sync-b")
    assert git(public, "rev-parse", f"{first.branch}^2") == git(
        public, "rev-parse", f"{second.branch}^2"
    )


def test_state_file_records_tip_allowlist_and_map(private: Path, public: Path) -> None:
    result = run(private, public)
    state = json.loads(git(public, "show", f"{result.branch}:{sync.STATE_PATH}"))
    assert state["source_tip"] == git(private, "rev-parse", "origin/main")
    assert state["allowlist"] == ["LICENSE", "pkg/core.py"]
    assert state["commit_map"][state["source_tip"]] == git(
        public, "rev-parse", f"{result.branch}^2"
    )


def test_incremental_sync_writes_only_new_commits(private: Path, public: Path) -> None:
    first = run(private, public)
    published_tip = git(public, "rev-parse", f"{first.branch}^2")
    merge_sync_branch(public, first.branch)
    commit(private, "Add module", {"pkg/extra.py": "y = 1\n"}, "1700000010 +0000")
    commit(private, "Tweak module", {"pkg/extra.py": "y = 2\n"}, "1700000011 +0000")
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")

    second = run(private, public, branch="sync-2", base="base-after-sync-1")
    assert second.status == "synced"
    assert second.written == 2
    new = git(public, "rev-list", "--reverse", f"upstream/main..{second.branch}^2")
    first_new = new.splitlines()[0]
    assert git(public, "rev-parse", f"{first_new}^") == published_tip


def test_deletion_after_first_sync_propagates(private: Path, public: Path) -> None:
    first = run(private, public)
    old_published = git(public, "rev-list", f"upstream/main..{first.branch}^2")
    merge_sync_branch(public, first.branch)
    commit(private, "Drop LICENSE", {"LICENSE": None}, "1700000010 +0000")
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")

    second = run(private, public, branch="sync-2", base="base-after-sync-1")
    subjects = git(public, "log", "--format=%s", f"upstream/main..{second.branch}^2")
    assert subjects.splitlines() == ["Drop LICENSE"]
    still_there = git(public, "rev-list", f"{first.branch}^2")
    for sha in old_published.splitlines():
        assert sha in still_there


def test_readme_only_commits_are_nothing_to_sync(private: Path, public: Path) -> None:
    first = run(private, public)
    merge_sync_branch(public, first.branch)
    commit(private, "Docs only", {"README.md": "# Changed\n"}, "1700000010 +0000")
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")

    second = run(private, public, branch="sync-2", base="base-after-sync-1")
    assert second.status == "nothing-to-sync"
    assert "# Changed" in second.readme_diff
    assert git(public, "branch", "--list", "sync-2") == ""


def test_no_new_commits_is_nothing_to_sync(private: Path, public: Path) -> None:
    first = run(private, public)
    merge_sync_branch(public, first.branch)
    second = run(private, public, branch="sync-2", base="base-after-sync-1")
    assert second.status == "nothing-to-sync"
    assert git(public, "branch", "--list", "sync-2") == ""


def test_force_pushed_source_stops(private: Path, public: Path) -> None:
    first = run(private, public)
    merge_sync_branch(public, first.branch)
    git(private, "reset", "-q", "--hard", "HEAD~1")
    commit(private, "Rewritten history", {"pkg/core.py": "x = 9\n"}, "1700000020 +0000")
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop, match="force-push"):
        run(private, public, branch="sync-2", base="base-after-sync-1")


def test_degenerate_merge_is_dropped_but_pr_merge_is_kept(
    tmp_path: Path, public: Path
) -> None:
    repo = init_repo(tmp_path / "merges")
    commit(repo, "Root", {"pkg/core.py": "x = 1\n"}, "1700000001 +0000")
    # A side branch that only touches a path the filter removes, merged back.
    git(repo, "switch", "-q", "-c", "side")
    commit(repo, "Side scratch", {"scratch.ipynb": "{}"}, "1700000002 +0000")
    git(repo, "switch", "-q", "main")
    commit(repo, "Main work", {"pkg/core.py": "x = 2\n"}, "1700000003 +0000")
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge side", "side")
    git(repo, "rm", "-q", "scratch.ipynb")
    git(repo, "commit", "-q", "-m", "Clean scratch")
    # A GitHub-style no-ff PR merge whose base did not move.
    git(repo, "switch", "-q", "-c", "feature")
    commit(repo, "Feature work", {"pkg/feature.py": "f = 1\n"}, "1700000005 +0000")
    git(repo, "switch", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #1", "feature")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")

    result = run(repo, public)
    subjects = git(public, "log", "--format=%s", f"upstream/main..{result.branch}^2")
    assert "Merge side" not in subjects.splitlines()
    assert subjects.splitlines()[0] == "Merge pull request #1"
    parents = git(public, "log", "-1", "--format=%P", f"{result.branch}^2").split()
    assert len(parents) == 2


# --- U2: overlay merge, scan, reports -------------------------------------


def test_overlay_tree_keeps_public_owned_files_and_drops_public_only_code(
    private: Path, public: Path
) -> None:
    result = run(private, public)
    tree = set(git(public, "ls-tree", "-r", "--name-only", result.branch).splitlines())
    assert tree == {
        "LICENSE",
        "README.md",
        "SELDON.png",
        "pkg/core.py",
        "tools/tfm_sync/__init__.py",
        sync.STATE_PATH,
    }
    assert git(public, "show", f"{result.branch}:LICENSE") == "BSD-3"
    assert git(public, "show", f"{result.branch}:README.md") == "# SELDON"
    assert git(public, "show", f"{result.branch}:pkg/core.py") == "x = 2"
    assert "plot_realtime.py" in result.deleted_paths


def test_sync_branch_joins_both_histories(private: Path, public: Path) -> None:
    result = run(private, public)
    parents = git(public, "log", "-1", "--format=%P", result.branch).split()
    assert parents[0] == git(public, "rev-parse", "base")
    assert len(parents) == 2


def test_leak_in_private_commit_blocks_and_creates_no_branch(
    private: Path, public: Path
) -> None:
    commit(private, "Oops", {"pkg/notes.md": f"see {FAKE_PRIVATE_CLONE}\n"})
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop, match="scan"):
        run(private, public)
    assert git(public, "branch", "--list", "sync-1") == ""


def test_leak_in_public_owned_file_on_base_blocks(private: Path, public: Path) -> None:
    commit(public, "Bad README", {"README.md": f"# SELDON\n{FAKE_FINGERPRINT}\n"})
    with pytest.raises(sync.SyncStop, match="scan"):
        run(private, public)


def test_leak_only_in_binary_blob_is_caught(private: Path, public: Path) -> None:
    payload = b"\x00\x01" + FAKE_PRIVATE_CLONE.encode() + b"\xff"
    commit(private, "Add data", {"pkg/data.bin": payload})
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop, match="scan"):
        run(private, public)


def test_leak_only_in_commit_message_is_caught(private: Path, public: Path) -> None:
    commit(private, f"Mirror of {FAKE_PRIVATE_CLONE}", {"pkg/core.py": "x = 3\n"})
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop, match="scan"):
        run(private, public)


def test_standalone_scan_reports_hits(private: Path, public: Path) -> None:
    result = run(private, public)
    assert sync.scan_branch(public, "upstream/main", result.branch) == []
    git(public, "switch", "-q", result.branch)
    commit(public, "Manual README edit", {"README.md": f"{FAKE_PRIVATE_CLONE}\n"})
    hits = sync.scan_branch(public, "upstream/main", result.branch)
    assert hits and hits[0].pattern == "private clone URL"


def test_squash_merge_stops_next_sync(private: Path, public: Path) -> None:
    first = run(private, public)
    git(public, "switch", "-q", "main")
    git(public, "merge", "-q", "--squash", first.branch)
    git(public, "commit", "-q", "-m", "Squashed sync")
    git(public, "update-ref", "refs/remotes/upstream/main", "main")
    git(public, "switch", "-q", "-c", "base2")
    commit(private, "More", {"pkg/core.py": "x = 5\n"}, "1700000030 +0000")
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop, match="squash or rebase"):
        run(private, public, branch="sync-2", base="base2")


def test_edits_and_additions_on_public_main_are_reported(
    private: Path, public: Path
) -> None:
    first = run(private, public)
    merge_sync_branch(public, first.branch)
    git(public, "switch", "-q", "main")
    commit(
        public,
        "Direct edits",
        {"pkg/core.py": "x = 100\n", "CONTRIBUTING.md": "Be nice\n"},
        "1700000040 +0000",
    )
    git(public, "update-ref", "refs/remotes/upstream/main", "main")
    git(public, "switch", "-q", "-c", "base2")
    commit(private, "More", {"pkg/core.py": "x = 6\n"}, "1700000041 +0000")
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")

    second = run(private, public, branch="sync-2", base="base2")
    assert "pkg/core.py" in second.edited_paths
    assert "CONTRIBUTING.md" in second.deleted_paths
    assert git(public, "show", f"{second.branch}:pkg/core.py") == "x = 6"


def test_existing_branch_name_is_refused(private: Path, public: Path) -> None:
    git(public, "branch", "sync-1")
    with pytest.raises(sync.SyncStop, match="already exists"):
        run(private, public)


def test_stopped_run_leaves_public_checkout_unchanged(
    private: Path, public: Path
) -> None:
    before = git(public, "for-each-ref", "--format=%(refname) %(objectname)")
    commit(private, "Oops", {"pkg/notes.md": f"{FAKE_FINGERPRINT}\n"})
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop):
        run(private, public)
    after = git(public, "for-each-ref", "--format=%(refname) %(objectname)")
    assert before == after


def test_private_objects_do_not_reach_public_checkout(
    private: Path, public: Path
) -> None:
    ckpt_blob = git(private, "rev-parse", "HEAD~2:logs/last.ckpt")
    run(private, public)
    probe = subprocess.run(
        ["git", "-C", str(public), "cat-file", "-e", ckpt_blob],
        capture_output=True,
    )
    assert probe.returncode != 0


# --- Review follow-ups: scan coverage, restored deletions, CLI ------------

# One leak-shaped string per scan pattern, each built from parts.
PATTERN_FIXTURES = {
    "private clone URL": "git" + "@github.com:" + "example/repo.git",
    "ssh clone URL": "git" + "+ssh://" + "example.org/repo",
    "private repo URL": "https://" + "github.com/" + "skai-institute/" + "seldon-dsb",
    "SSH key fingerprint": FAKE_FINGERPRINT,
    "ssh directory path": "~/." + "ssh/" + "config",
}


def test_every_scan_pattern_has_a_fixture() -> None:
    assert {name for name, _ in sync.SCAN_PATTERNS} == set(PATTERN_FIXTURES)


@pytest.mark.parametrize("name", sorted(PATTERN_FIXTURES))
def test_each_scan_pattern_is_caught(public: Path, name: str) -> None:
    commit(public, "Leaky note", {"notes.md": PATTERN_FIXTURES[name] + "\n"})
    hits = sync.scan_branch(public, "upstream/main", "base")
    assert name in {hit.pattern for hit in hits}


def test_private_repo_url_match_ignores_case(public: Path) -> None:
    lowered = (
        "https://" + "github.com/" + "skai-institute/" + "transient-foundation-model"
    )
    commit(public, "Lowercase link", {"notes.md": lowered + "\n"})
    hits = sync.scan_branch(public, "upstream/main", "base")
    assert [hit.pattern for hit in hits] == ["private repo URL"]


def test_leak_only_in_path_name_is_caught(private: Path, public: Path) -> None:
    commit(private, "Add config", {"pkg/." + "ssh/" + "config": "clean\n"})
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop, match="in path pkg/"):
        run(private, public)


def test_leak_only_in_author_header_is_caught(private: Path, public: Path) -> None:
    (private / "pkg" / "core.py").write_text("x = 7\n")
    git(private, "add", "pkg/core.py")
    git(
        private,
        "commit",
        "-q",
        "-m",
        "Clean message",
        f"--author=Leaky <{FAKE_PRIVATE_CLONE}>",
    )
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(sync.SyncStop, match="commit headers"):
        run(private, public)


def test_imported_file_deleted_on_public_main_is_reported(
    private: Path, public: Path
) -> None:
    first = run(private, public)
    merge_sync_branch(public, first.branch)
    git(public, "switch", "-q", "main")
    commit(public, "Remove core", {"pkg/core.py": None}, "1700000050 +0000")
    git(public, "update-ref", "refs/remotes/upstream/main", "main")
    git(public, "switch", "-q", "-c", "base2")
    commit(private, "More", {"pkg/extra.py": "y = 1\n"}, "1700000051 +0000")
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")

    second = run(private, public, branch="sync-2", base="base2")
    assert "pkg/core.py" in second.edited_paths
    assert git(public, "show", f"{second.branch}:pkg/core.py") == "x = 2"


def _cli(*args: str) -> int:
    from tools.tfm_sync.__main__ import main

    return main(list(args))


def test_cli_exit_codes(private: Path, public: Path, tmp_path: Path) -> None:
    common = ["--source", str(private), "--public", str(public), "--base", "base"]
    assert _cli("sync", *common, "--branch", "sync-ok") == 0
    assert _cli("scan", "--public", str(public), "--branch", "sync-ok") == 0
    commit(public, "Leak", {"README.md": FAKE_FINGERPRINT + "\n"})
    assert _cli("scan", "--public", str(public), "--branch", "base") == 2
    assert _cli("sync", *common, "--branch", "sync-leak") == 2
    missing = ["--source", str(tmp_path / "nope"), "--public", str(public)]
    assert _cli("sync", *missing, "--base", "base", "--branch", "x") == 2


def test_planning_docs_are_never_imported(private: Path, public: Path) -> None:
    commit(
        private,
        "Add planning docs",
        {
            "docs/plans/2026-01-01-plan.md": "plan\n",
            "docs/brainstorms/2026-01-01-requirements.md": "ideas\n",
            "docs/guide.md": "kept\n",
        },
    )
    git(private, "update-ref", "refs/remotes/origin/main", "HEAD")
    result = run(private, public)
    touched = paths_ever_touched(public, f"upstream/main..{result.branch}")
    assert not any(p.startswith(("docs/plans/", "docs/brainstorms/")) for p in touched)
    assert "docs/guide.md" in touched
    state = json.loads(git(public, "show", f"{result.branch}:{sync.STATE_PATH}"))
    assert not any(p.startswith("docs/plans/") for p in state["allowlist"])
