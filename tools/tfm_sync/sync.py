"""Sync the private development repository's history into this public repo.

The private development repository (TFM) is where SELDON is developed. This
module rewrites TFM ``main``'s history so that only allowed paths survive,
keeps every author, committer, date and message unchanged, and joins the result
to public ``main`` through a merge commit on a new local branch. The operator
pushes that branch to the machine-account fork and opens a pull request.

All rewriting happens in a throwaway bare repository, so private objects that
the filter removes (old checkpoints, old README blobs) never enter the public
checkout. Before the branch is fetched back into the public checkout, every
object and commit message the branch adds is scanned for private URLs and SSH
details; any hit stops the run.

State lives in ``tools/tfm_sync/state.json`` on public ``main``: the last
synced TFM tip, the cumulative allowlist of paths, and the private-to-public
commit map that keeps rewritten SHAs stable from one sync to the next.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = "tools/tfm_sync/state.json"

# Paths the public side owns: always taken from the public base branch, never
# from TFM. Entries ending in "/" are directory prefixes.
PUBLIC_OWNED = (
    "README.md",
    "LICENSE",
    "SELDONI.pdf",
    "SELDON.png",
    "RELEASING.md",
    "tools/tfm_sync/",
    "tests/test_tfm_sync.py",
)

# Paths never imported from TFM history, at any sync; entries ending in "/"
# are directory prefixes. TFM's old README versions contain private access
# details, and its planning documents stay private at the lead's request.
NEVER_IMPORTED = (
    "README.md",
    "docs/brainstorms/",
    "docs/plans/",
)

# Leak patterns. Each literal is split so this file never matches itself.
SCAN_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("private clone URL", re.compile(rb"git" + rb"@github\.com:")),
    ("ssh clone URL", re.compile(rb"git" + rb"\+ssh://")),
    (
        "private repo URL",
        re.compile(
            rb"github\.com[:/]skai-institute/(?:Transient-Foundation-Model|seldon-dsb)",
            re.IGNORECASE,
        ),
    ),
    ("SSH key fingerprint", re.compile(rb"SHA256:[A-Za-z0-9+/]{43}")),
    ("ssh directory path", re.compile(rb"\." + rb"ssh/")),
)

DEFAULT_SOURCE_REF = "origin/main"
DEFAULT_PUBLIC_REF = "upstream/main"

_SCRATCH_PUBLIC = "refs/sync/public"
_SCRATCH_BASE = "refs/sync/base"
_SCRATCH_SOURCE = "refs/sync/source"
_SCRATCH_OUT = "refs/heads/out"


class SyncStop(Exception):
    """A guard or the leak scan stopped the run; nothing was written."""


@dataclass
class ScanHit:
    """One leak-pattern match found by the scan.

    Attributes:
        pattern: Human-readable pattern name.
        object_id: SHA of the blob or commit that matched.
        where: The blob's path, or ``"commit message"``.
    """

    pattern: str
    object_id: str
    where: str


@dataclass
class SyncState:
    """Persistent sync state stored at ``STATE_PATH`` on public main.

    Attributes:
        source_tip: Last synced TFM commit, or ``None`` before the first sync.
        allowlist: Sorted cumulative list of paths allowed into public history.
        commit_map: TFM commit SHA -> rewritten public SHA (``None`` when a
            dropped commit has no surviving ancestor).
    """

    source_tip: str | None = None
    allowlist: list[str] = field(default_factory=list)
    commit_map: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> SyncState:
        """Parse a state file.

        Args:
            text: Contents of ``state.json``.

        Returns:
            The parsed state.
        """
        data = json.loads(text)
        return cls(
            source_tip=data["source_tip"],
            allowlist=list(data["allowlist"]),
            commit_map=dict(data["commit_map"]),
        )

    def to_json(self) -> str:
        """Serialize deterministically so state diffs stay reviewable.

        Returns:
            JSON text with sorted keys and a trailing newline.
        """
        data = {
            "allowlist": sorted(self.allowlist),
            "commit_map": self.commit_map,
            "source_tip": self.source_tip,
        }
        return json.dumps(data, indent=2, sort_keys=True) + "\n"


@dataclass
class SyncResult:
    """Outcome of one sync run.

    Attributes:
        status: ``"synced"`` or ``"nothing-to-sync"``.
        branch: The created branch, or ``None`` when nothing was synced.
        written: Rewritten commits created by this run.
        dropped: TFM commits the filter dropped in this run.
        readme_diff: TFM ``README.md`` diff since the last synced tip.
        edited_paths: Imported paths public main edited or deleted since the
            last sync; the TFM version overwrites or restores them.
        deleted_paths: Base-branch paths that are neither public-owned nor in
            the new TFM tree; the sync merge deletes them.
    """

    status: str
    branch: str | None = None
    written: int = 0
    dropped: int = 0
    readme_diff: str = ""
    edited_paths: list[str] = field(default_factory=list)
    deleted_paths: list[str] = field(default_factory=list)


class Git:
    """Minimal git runner bound to one repository."""

    def __init__(self, repo: Path, bare: bool = False) -> None:
        """Bind to a repository.

        Args:
            repo: Working tree path, or the git dir itself when ``bare``.
            bare: Whether ``repo`` is a bare repository.
        """
        self.repo = repo
        self._prefix = (
            ["git", f"--git-dir={repo}"] if bare else ["git", "-C", str(repo)]
        )

    def run(
        self,
        *args: str,
        input_bytes: bytes | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> bytes:
        """Run a git command and return its stdout.

        Args:
            *args: Git arguments.
            input_bytes: Optional stdin.
            env: Extra environment variables.
            check: Raise on a non-zero exit.

        Returns:
            Raw stdout bytes.

        Raises:
            subprocess.CalledProcessError: If ``check`` and git fails.
        """
        full_env = dict(os.environ, **(env or {}))
        result = subprocess.run(
            [*self._prefix, *args],
            input=input_bytes,
            capture_output=True,
            env=full_env,
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        return result.stdout

    def text(
        self,
        *args: str,
        input_bytes: bytes | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> str:
        """Run a git command and return stripped text stdout; see ``run``."""
        out = self.run(*args, input_bytes=input_bytes, env=env, check=check)
        return out.decode().strip()

    def ok(self, *args: str) -> bool:
        """Return whether a git command exits zero."""
        return (
            subprocess.run([*self._prefix, *args], capture_output=True).returncode == 0
        )

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Return whether ``ancestor`` is an ancestor of ``descendant``."""
        return self.ok("merge-base", "--is-ancestor", ancestor, descendant)

    def tree_entries(self, rev: str) -> dict[str, tuple[str, str]]:
        """List every file in a commit's tree.

        Args:
            rev: Commit or tree to list.

        Returns:
            Path -> (mode, object SHA).
        """
        out = self.run("ls-tree", "-r", "-z", "--full-tree", rev)
        entries: dict[str, tuple[str, str]] = {}
        for record in out.split(b"\0"):
            if not record:
                continue
            meta, path = record.split(b"\t", 1)
            mode, _type, sha = meta.decode().split()
            entries[path.decode()] = (mode, sha)
        return entries

    def write_tree(self, entries: dict[str, tuple[str, str]]) -> str:
        """Write a tree object from path entries using a temporary index.

        Args:
            entries: Path -> (mode, object SHA).

        Returns:
            The new tree's SHA.
        """
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
            self.run("read-tree", "--empty", env=env)
            info = b"".join(
                f"{mode} {sha}\t{path}".encode() + b"\0"
                for path, (mode, sha) in sorted(entries.items())
            )
            if info:
                self.run(
                    "update-index", "-z", "--index-info", input_bytes=info, env=env
                )
            return self.text("write-tree", env=env)


def _matches(path: str, entries: tuple[str, ...]) -> bool:
    """Return whether ``path`` equals an entry or lies under a ``/`` prefix."""
    return any(
        path.startswith(entry) if entry.endswith("/") else path == entry
        for entry in entries
    )


def is_public_owned(path: str) -> bool:
    """Return whether the public side owns ``path``.

    Args:
        path: Repository-relative path.

    Returns:
        True when the path comes from the public base branch, never from TFM.
    """
    return _matches(path, PUBLIC_OWNED)


def compute_allowlist(tip_paths: set[str], previous: list[str]) -> set[str]:
    """Compute the paths allowed into rewritten commits.

    Args:
        tip_paths: Paths present at the current TFM tip.
        previous: The allowlist recorded by the last sync (empty at first).

    Returns:
        Tip paths plus previously allowed paths, minus never-imported paths.
    """
    return {
        path for path in tip_paths | set(previous) if not _matches(path, NEVER_IMPORTED)
    }


def _commit_message(raw: bytes) -> bytes:
    """Return the exact message bytes of a raw commit object."""
    return raw.split(b"\n\n", 1)[1] if b"\n\n" in raw else b""


def _commit_meta(git: Git, sha: str) -> tuple[dict[str, str], bytes]:
    """Read a commit's identity, dates and raw message.

    Args:
        git: Repository holding the commit.
        sha: Commit SHA.

    Returns:
        Environment variables for ``commit-tree`` and the exact message bytes.
    """
    fmt = "%an%x00%ae%x00%ad%x00%cn%x00%ce%x00%cd"
    fields = git.run(
        "-c",
        "log.showSignature=false",
        "log",
        "-1",
        f"--format={fmt}",
        "--date=raw",
        sha,
    ).decode()
    an, ae, ad, cn, ce, cd = fields.rstrip("\n").split("\0")
    env = {
        "GIT_AUTHOR_NAME": an,
        "GIT_AUTHOR_EMAIL": ae,
        "GIT_AUTHOR_DATE": ad,
        "GIT_COMMITTER_NAME": cn,
        "GIT_COMMITTER_EMAIL": ce,
        "GIT_COMMITTER_DATE": cd,
    }
    return env, _commit_message(git.run("cat-file", "commit", sha))


def rewrite_history(
    git: Git, source_ref: str, state: SyncState, allowlist: set[str]
) -> tuple[dict[str, str | None], int, int]:
    """Rewrite TFM commits that are not yet in the commit map.

    Commits are walked oldest first. Each tree is filtered to ``allowlist``,
    parents are mapped through the commit map, and a commit whose filtered
    tree equals its single remaining parent's tree is dropped.

    Args:
        git: Scratch repository holding both histories.
        source_ref: Ref of the TFM tip to sync.
        state: Current sync state; its commit map is not modified.
        allowlist: Paths allowed into rewritten commits.

    Returns:
        The extended commit map, commits written, and commits dropped.
    """
    commit_map = dict(state.commit_map)
    written = dropped = 0
    order = git.text("rev-list", "--topo-order", "--reverse", source_ref).split()
    for sha in order:
        if sha in commit_map:
            continue
        source_parents = git.text("log", "-1", "--format=%P", sha).split()
        mapped: list[str] = []
        for parent in source_parents:
            target = commit_map.get(parent)
            if target and target not in mapped:
                mapped.append(target)
        related = len(source_parents) > 1 and any(
            git.is_ancestor(a, b)
            for a in source_parents
            for b in source_parents
            if a != b
        )
        if len(mapped) > 1 and not related:
            mapped = [
                p
                for p in mapped
                if not any(q != p and git.is_ancestor(p, q) for q in mapped)
            ]
        entries = {
            path: entry
            for path, entry in git.tree_entries(sha).items()
            if path in allowlist
        }
        tree = git.write_tree(entries)
        if len(mapped) <= 1:
            parent = mapped[0] if mapped else None
            if parent:
                drop = tree == git.text("rev-parse", f"{parent}^{{tree}}")
            else:
                drop = not entries
            if drop:
                commit_map[sha] = parent
                dropped += 1
                continue
        env, message = _commit_meta(git, sha)
        args = ["commit-tree", tree]
        for parent in mapped:
            args += ["-p", parent]
        commit_map[sha] = git.text(*args, input_bytes=message, env=env)
        written += 1
    return commit_map, written, dropped


def scan(git: Git, base: str, head: str) -> list[ScanHit]:
    """Scan everything that ``head`` adds on top of ``base``.

    Blobs are scanned byte-for-byte, so binary files and content introduced
    only by merge commits are covered. Every path name and the complete raw
    commit objects (author and committer headers as well as messages) are
    scanned too.

    Args:
        git: Repository holding both revisions.
        base: Revision already public (public main).
        head: Branch head about to be pushed.

    Returns:
        Every pattern match, in object order then pattern order.
    """
    hits: list[ScanHit] = []
    listing = git.text("rev-list", "--objects", f"{base}..{head}")
    paths: dict[str, str] = {}
    for line in listing.splitlines():
        sha, _, path = line.partition(" ")
        paths.setdefault(sha, path)
        hits.extend(
            ScanHit(name, sha, f"path {path}")
            for name, pattern in SCAN_PATTERNS
            if pattern.search(path.encode())
        )
    if paths:
        batch = git.run(
            "cat-file", "--batch", input_bytes="\n".join(paths).encode() + b"\n"
        )
        pos = 0
        while pos < len(batch):
            header_end = batch.index(b"\n", pos)
            sha, obj_type, size = batch[pos:header_end].decode().split()
            body = batch[header_end + 1 : header_end + 1 + int(size)]
            pos = header_end + 1 + int(size) + 1
            if obj_type == "blob":
                haystack, where = body, paths[sha]
            elif obj_type == "commit":
                haystack, where = body, "commit headers or message"
            else:
                continue
            hits.extend(
                ScanHit(name, sha, where)
                for name, pattern in SCAN_PATTERNS
                if pattern.search(haystack)
            )
    return hits


def scan_branch(public: Path, public_ref: str, branch: str) -> list[ScanHit]:
    """Scan a branch in the public checkout before it is pushed.

    Args:
        public: Public repository checkout.
        public_ref: Ref of public main, e.g. ``upstream/main``.
        branch: Branch about to be pushed.

    Returns:
        Every pattern match; empty means the push is allowed.
    """
    return scan(Git(public), public_ref, branch)


def _read_state(git: Git, ref: str) -> SyncState:
    """Load ``state.json`` from ``ref``, or an empty state when absent."""
    blob = git.run("show", f"{ref}:{STATE_PATH}", check=False)
    if not blob.strip():
        return SyncState()
    return SyncState.from_json(blob.decode())


def _full_ref(git: Git, ref: str) -> str:
    """Resolve a short ref name to its full name, or raise ``SyncStop``."""
    name = git.text("rev-parse", "--symbolic-full-name", ref, check=False)
    if not name.startswith("refs/"):
        raise SyncStop(f"cannot resolve ref {ref!r} in {git.repo}")
    return name


def _operator_identity(public: Git) -> dict[str, str]:
    """Identity for the sync merge commit: public checkout config, else env."""
    name = public.text("config", "user.name", check=False)
    email = public.text("config", "user.email", check=False)
    name = name or os.environ.get("GIT_AUTHOR_NAME", "tfm-sync")
    email = email or os.environ.get("GIT_AUTHOR_EMAIL", "tfm-sync@localhost")
    return {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }


def run_sync(
    source: Path,
    public: Path,
    base: str,
    branch: str,
    source_ref: str = DEFAULT_SOURCE_REF,
    public_ref: str = DEFAULT_PUBLIC_REF,
) -> SyncResult:
    """Run one sync and, on success, create ``branch`` in the public checkout.

    Args:
        source: Checkout of the private development repository.
        public: Checkout of this public repository.
        base: Public branch to merge onto (public main plus public-owned edits).
        branch: Name of the sync branch to create.
        source_ref: TFM ref to sync; the worktree and HEAD are never read.
        public_ref: Ref of public main, which holds ``state.json``.

    Returns:
        The run's outcome and reports.

    Raises:
        SyncStop: When a guard or the leak scan stops the run.
    """
    public_git = Git(public)
    if public_git.ok("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"):
        raise SyncStop(f"branch {branch!r} already exists in {public}")
    source_full = _full_ref(Git(source), source_ref)
    public_full = _full_ref(public_git, public_ref)
    base_full = _full_ref(public_git, base)

    with tempfile.TemporaryDirectory(
        prefix="tfm-sync-", ignore_cleanup_errors=True
    ) as scratch_dir:
        scratch = Git(Path(scratch_dir), bare=True)
        scratch.run("init", "--bare", "-q")
        scratch.run(
            "fetch",
            "-q",
            "--no-tags",
            str(public),
            f"{public_full}:{_SCRATCH_PUBLIC}",
            f"{base_full}:{_SCRATCH_BASE}",
        )
        scratch.run(
            "fetch", "-q", "--no-tags", str(source), f"{source_full}:{_SCRATCH_SOURCE}"
        )
        return _sync_in_scratch(scratch, public_git, branch)


def _sync_in_scratch(scratch: Git, public_git: Git, branch: str) -> SyncResult:
    """Rewrite, merge, scan and fetch back; see ``run_sync``."""
    state = _read_state(scratch, _SCRATCH_PUBLIC)
    new_tip = scratch.text("rev-parse", _SCRATCH_SOURCE)
    previous_public: str | None = None
    readme_diff = ""
    if state.source_tip:
        if not scratch.is_ancestor(state.source_tip, _SCRATCH_SOURCE):
            raise SyncStop(
                "recorded TFM tip is not an ancestor of the source ref; "
                "TFM main was force-pushed, so the sync cannot continue"
            )
        previous_public = state.commit_map.get(state.source_tip)
        if previous_public and not scratch.is_ancestor(
            previous_public, _SCRATCH_PUBLIC
        ):
            raise SyncStop(
                "the last synced commit is not in public main's history; "
                "the previous sync PR was merged by squash or rebase"
            )
        readme_diff = scratch.text("diff", state.source_tip, new_tip, "--", "README.md")

    if new_tip == state.source_tip:
        logger.info("nothing to sync: no new TFM commits")
        return SyncResult(status="nothing-to-sync", readme_diff=readme_diff)

    tip_paths = set(scratch.tree_entries(_SCRATCH_SOURCE))
    allowlist = compute_allowlist(tip_paths, state.allowlist)
    commit_map, written, dropped = rewrite_history(
        scratch, _SCRATCH_SOURCE, state, allowlist
    )
    public_tip = commit_map[new_tip]
    if public_tip is None or public_tip == previous_public:
        logger.info("nothing to sync: every new TFM commit was filtered out")
        return SyncResult(
            status="nothing-to-sync", dropped=dropped, readme_diff=readme_diff
        )

    new_state = SyncState(
        source_tip=new_tip, allowlist=sorted(allowlist), commit_map=commit_map
    )
    base_entries = scratch.tree_entries(_SCRATCH_BASE)
    tip_entries = scratch.tree_entries(public_tip)
    merged = {p: e for p, e in tip_entries.items() if not is_public_owned(p)}
    merged.update(
        {
            p: e
            for p, e in base_entries.items()
            if is_public_owned(p) and p != STATE_PATH
        }
    )
    state_blob = scratch.text(
        "hash-object", "-w", "--stdin", input_bytes=new_state.to_json().encode()
    )
    merged[STATE_PATH] = ("100644", state_blob)

    deleted = sorted(p for p in base_entries if p not in merged)
    edited: list[str] = []
    if previous_public:
        previous_entries = scratch.tree_entries(previous_public)
        edited = sorted(
            p
            for p, entry in previous_entries.items()
            if not is_public_owned(p) and p in merged and base_entries.get(p) != entry
        )

    tree = scratch.write_tree(merged)
    message = (
        f"chore: sync TFM main up to {new_tip[:7]}\n\n"
        f"Imports {written} rewritten commit(s) from the private development\n"
        f"repository's main branch (source tip {new_tip}).\n"
    ).encode()
    head = scratch.text(
        "commit-tree",
        tree,
        "-p",
        scratch.text("rev-parse", _SCRATCH_BASE),
        "-p",
        public_tip,
        input_bytes=message,
        env=_operator_identity(public_git),
    )

    hits = scan(scratch, _SCRATCH_PUBLIC, head)
    if hits:
        details = "; ".join(
            f"{h.pattern} in {h.where} ({h.object_id[:10]})" for h in hits
        )
        raise SyncStop(f"leak scan found {len(hits)} hit(s): {details}")

    scratch.run("update-ref", _SCRATCH_OUT, head)
    public_git.run(
        "fetch",
        "-q",
        "--no-tags",
        str(scratch.repo),
        f"{_SCRATCH_OUT}:refs/heads/{branch}",
    )
    logger.info("created %s: %d written, %d dropped", branch, written, dropped)
    return SyncResult(
        status="synced",
        branch=branch,
        written=written,
        dropped=dropped,
        readme_diff=readme_diff,
        edited_paths=edited,
        deleted_paths=deleted,
    )
