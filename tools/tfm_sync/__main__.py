"""Command-line entry point: ``python -m tools.tfm_sync {sync,scan} ...``.

Run from the public repository root. See ``tools/tfm_sync/README.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tools.tfm_sync import sync

logger = logging.getLogger("tools.tfm_sync")

EXIT_OK = 0
EXIT_STOPPED = 2


def _print_report(result: sync.SyncResult) -> None:
    """Print the operator-facing reports for a sync run."""
    print(f"status: {result.status}")
    print(f"commits written: {result.written}, dropped: {result.dropped}")
    if result.branch:
        print(f"branch: {result.branch}")
    for label, paths in (
        (
            "imported paths edited on public main (TFM version wins)",
            result.edited_paths,
        ),
        ("paths the sync merge deletes", result.deleted_paths),
    ):
        print(f"{label}: {len(paths)}")
        for path in paths:
            print(f"  {path}")
    print("TFM README.md changes since the last sync (reconcile by hand):")
    print(result.readme_diff or "  (none)")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a subcommand.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv``.

    Returns:
        Process exit code: 0 on success or nothing to sync, 2 when stopped.
    """
    parser = argparse.ArgumentParser(prog="python -m tools.tfm_sync")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("sync", help="rewrite new TFM commits onto a sync branch")
    run.add_argument("--source", type=Path, required=True, help="TFM checkout")
    run.add_argument("--source-ref", default=sync.DEFAULT_SOURCE_REF)
    run.add_argument("--public", type=Path, default=Path.cwd(), help="this checkout")
    run.add_argument("--public-ref", default=sync.DEFAULT_PUBLIC_REF)
    run.add_argument("--base", required=True, help="branch to merge onto")
    run.add_argument("--branch", required=True, help="sync branch to create")

    check = sub.add_parser("scan", help="scan a branch before pushing it")
    check.add_argument("--public", type=Path, default=Path.cwd())
    check.add_argument("--public-ref", default=sync.DEFAULT_PUBLIC_REF)
    check.add_argument("--branch", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "scan":
        hits = sync.scan_branch(args.public, args.public_ref, args.branch)
        for hit in hits:
            logger.error("%s in %s (%s)", hit.pattern, hit.where, hit.object_id)
        if hits:
            logger.error("scan failed: do not push %s", args.branch)
            return EXIT_STOPPED
        logger.info("scan clean: %s may be pushed", args.branch)
        return EXIT_OK

    try:
        result = sync.run_sync(
            source=args.source,
            public=args.public,
            base=args.base,
            branch=args.branch,
            source_ref=args.source_ref,
            public_ref=args.public_ref,
        )
    except sync.SyncStop as stop:
        logger.error("stopped: %s", stop)
        return EXIT_STOPPED
    _print_report(result)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
