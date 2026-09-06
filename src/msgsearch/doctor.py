"""Preflight checks for a new setup.

Every failure this reports is one somebody would otherwise meet as a confusing
error much later. Installing on an Intel-built Python fails with "No matching
distribution found for torch", which says nothing about architecture. A gated
model fails with a bare 401 that says "please log in" even when you already have.
An index built by a different model produces plausible, wrong results with no
error at all.

Each check therefore reports what is wrong *and* the command that fixes it.
"""

from __future__ import annotations

import platform
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import config

OK, WARN, FAIL = "ok", "warn", "fail"

_MARK = {OK: "✓", WARN: "!", FAIL: "✗"}

# The live database. Copying from it is fine; pointing the tool at it is not.
LIVE_DB = Path("~/Library/Messages/chat.db").expanduser()


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def check_python() -> Check:
    machine = platform.machine()
    version = platform.python_version()
    if machine != "arm64" and platform.system() == "Darwin":
        return Check(
            "python",
            FAIL,
            f"{version} on {machine} (Rosetta/Intel build)",
            "PyTorch publishes no x86_64 macOS wheels, and a Rosetta process cannot "
            "use the GPU. Use an arm64 Python 3.10-3.13 and recreate the venv.",
        )
    return Check("python", OK, f"{version} on {machine}")


def check_torch() -> Check:
    try:
        import torch
    except ImportError:
        return Check(
            "pytorch", FAIL, "not installed", "pip install -e . (in an arm64 venv)"
        )

    if torch.backends.mps.is_available():
        return Check("pytorch", OK, f"{torch.__version__}, Metal (MPS) available")
    return Check(
        "pytorch",
        WARN,
        f"{torch.__version__}, no GPU backend",
        "Indexing will run on the CPU and take considerably longer.",
    )


def check_database() -> Check:
    path = Path(config.DB_PATH).expanduser()

    if path.resolve() == LIVE_DB.resolve():
        return Check(
            "database",
            FAIL,
            f"pointed at the live database ({path})",
            "That file is live, locked by Messages, and irreplaceable. Take a "
            "snapshot instead: msgsearch sync",
        )

    if not path.exists():
        hint = "msgsearch sync   (needs Full Disk Access for the app running it)"
        return Check("database", FAIL, f"no database at {path}", hint)

    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        total, blobs = db.execute(
            "SELECT count(*), sum(text IS NULL AND attributedBody IS NOT NULL) "
            "FROM message"
        ).fetchone()
        db.close()
    except sqlite3.Error as error:
        return Check(
            "database",
            FAIL,
            f"cannot read {path}: {error}",
            "Grant your terminal Full Disk Access, or re-copy the file.",
        )

    size = path.stat().st_size / 1e6
    hidden = 100 * (blobs or 0) / total if total else 0
    return Check(
        "database",
        OK,
        f"{total:,} messages, {size:.0f} MB ({hidden:.0f}% need blob decoding)",
    )


def check_model() -> Check:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return Check("model", FAIL, "huggingface_hub not installed", "pip install -e .")

    name = config.EMBED_MODEL
    try:
        hf_hub_download(name, "config.json")
    except Exception as error:
        text = str(error)
        if "gated" in text.lower() or "401" in text:
            return Check(
                "model",
                FAIL,
                f"{name} is gated and this machine is not authorised",
                f"Accept the licence at https://huggingface.co/{name}, then run "
                "'hf auth login'. An expired token gives this same error, so log in "
                "again even if you have before. Or set "
                "MSGSEARCH_EMBED_MODEL=BAAI/bge-small-en-v1.5 to use an ungated model.",
            )
        return Check("model", WARN, f"{name}: {text.splitlines()[0][:80]}")
    return Check("model", OK, f"{name} reachable")


def check_index() -> Check:
    index_dir = Path(config.INDEX_DIR).expanduser()
    db_path = index_dir / "index.db"
    vec_path = index_dir / "vectors.npy"

    if not (db_path.exists() and vec_path.exists()):
        return Check("index", WARN, f"not built ({index_dir})", "msgsearch index")

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    meta = dict(db.execute("SELECT key, value FROM meta"))
    db.close()

    built_with = meta.get("embed_model", "?")
    if built_with != config.EMBED_MODEL:
        return Check(
            "index",
            FAIL,
            f"built with {built_with}, configured model is {config.EMBED_MODEL}",
            "Vectors from different models are not comparable. Either set "
            f"MSGSEARCH_EMBED_MODEL={built_with} or rebuild with 'msgsearch index'.",
        )

    indexed = int(meta.get("message_count", 0))
    detail = (
        f"{int(meta.get('window_count', 0)):,} windows, "
        f"{int(meta.get('passage_count', 0)):,} passages, "
        f"built {meta.get('built_at', '?')}"
    )

    fresh = _messages_available(meta.get("chat_filter") or None)
    if fresh is not None and fresh > indexed:
        return Check(
            "index",
            WARN,
            f"{detail} — {fresh - indexed:,} new messages since",
            "msgsearch index   (reuses existing embeddings; only new text is embedded)",
        )
    return Check("index", OK, detail)


def _messages_available(chat_identifier: str | None) -> int | None:
    """Count indexable messages currently in the database, or None if unreadable."""
    path = Path(config.DB_PATH).expanduser()
    if not path.exists():
        return None
    where = """
        m.associated_message_type = 0 AND m.item_type = 0 AND m.is_empty = 0
        AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL)
    """
    sql = f"""SELECT count(DISTINCT m.ROWID) FROM message m
              JOIN chat_message_join j ON j.message_id = m.ROWID
              WHERE {where}"""
    params: tuple = ()
    if chat_identifier:
        sql += " AND j.chat_id IN (SELECT ROWID FROM chat WHERE chat_identifier = ?)"
        params = (chat_identifier,)
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        count = db.execute(sql, params).fetchone()[0]
        db.close()
        return count
    except sqlite3.Error:
        return None


CHECKS = (check_python, check_torch, check_database, check_model, check_index)


def run() -> int:
    """Run every check, print a report, and return a process exit code."""
    print("msgsearch doctor\n")
    results = [check() for check in CHECKS]
    width = max(len(r.name) for r in results)

    for result in results:
        print(f"  {_MARK[result.status]} {result.name:<{width}}  {result.detail}")
        if result.fix:
            for line in _wrap(result.fix, width + 6):
                print(line)

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    print()
    if failures:
        print(f"{len(failures)} problem(s) to fix before this will work.")
        return 1
    if warnings:
        print("Ready, with warnings above.")
        return 0
    print("All good.")
    return 0


def _wrap(text: str, indent: int, width: int = 78) -> list[str]:
    import textwrap

    prefix = " " * indent
    return textwrap.wrap(
        text, width=width, initial_indent=prefix, subsequent_indent=prefix
    )


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
