"""Where the canonical memory store lives, and the one rule for finding it.

A connection is never built from constants in code. It comes from
``PERSONAL_RECALL_DATABASE_URL`` — the single variable this project reads — so that no password, no
host and no database name is ever written into a source file, a log line or a commit (§5, §36).
``docker-compose.postgres.yml`` documents a local development URL, but ``.env.example`` is where it
is spelled out as a placeholder and ``.env`` (git-ignored) is where it becomes real.

The store is deliberately **optional**. Everything in Personal Recall — import, index, retrieval,
the web UI — works with this module's functions never being called, because Phase 22A must not make
a database a prerequisite for asking a question (§22, §38).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The one environment variable that names the canonical store. Read through :func:`database_url`,
#: never directly, so "where did this connection come from" has a single answer.
DATABASE_URL_VAR = "PERSONAL_RECALL_DATABASE_URL"

#: Overrides the store's logical name (default: the account directory's name). Only used to make a
#: `--rebuild` prove it is aimed at the store it thinks it is (§37).
STORE_NAME_VAR = "PERSONAL_RECALL_STORE_NAME"


class StoreNotConfigured(RuntimeError):
    """No database URL is configured.

    Raised by the *explicit* store commands — ``--bootstrap-postgres`` and friends — and never by
    anything on the retrieval path. A user who has not set up PostgreSQL still has a working product;
    what they do not have is a silently-skipped database command that reported success (§38).
    """


@dataclass(frozen=True)
class StoreTarget:
    """A resolved destination: the URL to connect to, and what the store calls itself.

    ``url`` is intentionally the one field that must never be printed. :meth:`describe` is what goes
    into logs, and it carries the database name and host but not the credentials.
    """

    url: str
    store_name: str

    def describe(self) -> str:
        """A loggable identity for the target: host, port and database — never the password.

        Parsed rather than regex-stripped so that a URL this project cannot understand is reported as
        ``(unparsable url)`` instead of being echoed back in full. Echoing an unparsable URL is the
        one case where a "redacted" string would leak the very credential it meant to hide.
        """
        try:
            from urllib.parse import urlsplit

            parts = urlsplit(self.url)
        except ValueError:
            return f"store={self.store_name} (unparsable url)"
        if not parts.hostname:
            return f"store={self.store_name} (unparsable url)"
        host = f"{parts.hostname}:{parts.port}" if parts.port else parts.hostname
        return f"store={self.store_name} db={parts.path.lstrip('/') or '(none)'} host={host}"


def database_url(explicit: str | None = None, env: dict[str, str] | None = None) -> str | None:
    """The configured database URL, or ``None``. An explicit argument always wins.

    ``None`` rather than an exception: "no database is configured" is a legitimate state that the
    caller decides what to do about, and a function called during argument parsing should not be the
    thing that decides.
    """
    if explicit:
        return explicit.strip() or None
    source = os.environ if env is None else env
    value = str(source.get(DATABASE_URL_VAR) or "").strip()
    return value or None


def store_name_for(account_dir: Path, explicit: str | None = None) -> str:
    """The store's logical name: the account export directory's name, unless overridden."""
    if explicit and explicit.strip():
        return explicit.strip()
    override = str(os.environ.get(STORE_NAME_VAR) or "").strip()
    if override:
        return override
    return Path(account_dir).resolve().name


def target_for(
    account_dir: Path,
    *,
    url: str | None = None,
    store_name: str | None = None,
) -> StoreTarget:
    """Resolve a connection target, or raise :class:`StoreNotConfigured`."""
    resolved = database_url(url)
    if not resolved:
        raise StoreNotConfigured(
            f"no PostgreSQL URL configured; set {DATABASE_URL_VAR} (see .env.example), "
            "or pass --postgres-url"
        )
    return StoreTarget(url=resolved, store_name=store_name_for(account_dir, store_name))
