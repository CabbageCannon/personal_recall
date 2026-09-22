"""Start the Personal Recall web UI, on this machine only.

    python web.py
    python web.py --account data/real/account --port 8000

One page with one question box: ask, and you get the answer with the evidence cards behind it and
the caveats worth reading. The index is built **once**, here, before the first request — indexing
parses every export and embeds every session, so per-question work would make the page unusable.

The account export tree is the only input, and it is produced by ``export_account.py``. There is no
upload: ingestion stays a CLI step by design, so the web process never has to be trusted with a file
it did not already have.

Why the bind address is a constant rather than a flag: this serves real private chat to whoever can
reach the port, and a wildcard address would publish it to the whole local network. The tool has no
way to ask for one — see ``webapp.HOST``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import uvicorn  # noqa: E402

from webapp import DEFAULT_PORT, HOST, RecallState, create_app  # noqa: E402

#: Where ``export_account.py`` writes by default, and the only place the UI reads.
DEFAULT_ACCOUNT_DIR = BASE_DIR / "data" / "real" / "account"

#: Overrides the default account directory without a flag, for a shell that always points at the
#: same export tree. A flag still wins over it, so the override is never a surprise.
ACCOUNT_ENV_VAR = "PERSONAL_RECALL_ACCOUNT"


def server_options(port: int) -> dict[str, Any]:
    """The uvicorn keyword arguments. ``host`` is read from the constant and cannot be overridden."""
    return {"host": HOST, "port": port, "log_level": "info"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask your chat history from a local web page.")
    parser.add_argument(
        "--account",
        type=Path,
        default=None,
        help=f"the account export directory written by export_account.py "
        f"(default: ${ACCOUNT_ENV_VAR}, else {DEFAULT_ACCOUNT_DIR})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"the local port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="where the persistent retrieval index lives (default: beside the account export, "
        "git-ignored); the CLI and the acceptance runner use the same default",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="ignore a valid persistent index and rebuild it from the exports (~80 min on a real "
        "account); without it a valid index is loaded and nothing is embedded",
    )
    args = parser.parse_args()

    account_dir = args.account or Path(os.environ.get(ACCOUNT_ENV_VAR) or DEFAULT_ACCOUNT_DIR)

    state = RecallState(
        account_dir=account_dir, index_dir=args.index_dir, rebuild_index=args.rebuild_index
    )
    print(f"account  : {account_dir}")
    print("index    : loading the persistent index, or building it if it is not valid")

    started = perf_counter()
    state.load()

    if state.ready:
        report = state.report
        # `build_account_session` decided and measured this; the server only prints it.
        for line in state.index_status.lines():
            print(f"index    : {line}")
        print(
            f"index    : {report.conversations_imported} conversation(s), "
            f"{report.messages_kept} message(s) in {perf_counter() - started:.1f}s"
        )
        if report.partial:
            # No Data Loaded != No Memory Exists, said out loud rather than left in a status field.
            # The report owns the wording: there are two independent ways to be PARTIAL — a shard that
            # was never exported, and an export deliberately narrowed to a subset of conversations —
            # and only it knows which happened. Restating just the shard case here would announce
            # "0 message shard(s) were not exported" over a history that is missing 269 conversations.
            for line in report.lines():
                if line.startswith("WARNING"):
                    print(f"index    : {line}")
    else:
        print("index    : FAILED - the server still starts; /api/status reports why", file=sys.stderr)

    print(f"open     : http://{HOST}:{args.port}")
    uvicorn.run(create_app(state), **server_options(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
