#!/usr/bin/env python3
"""Stop hook: refuse to finish a retrieval change that made retrieval worse.

Must be launched with .venv/bin/python (settings.json does this). Under the
system python3, which is an x86_64 build, child processes inherit that
architecture and the arm64 numpy in the venv fails to load -- the bench then
exits non-zero and the check silently passes.

AGENTS.md asks an agent to justify retrieval changes with eval/bench.py. This
turns that request into a wall, because a request is something a model can
forget and a hook is not.

Fires only when a file that can change ranking was touched, and only when the
gold set actually has labels -- there is nothing to measure otherwise.

Exit 0 allows the stop. Exit 2 blocks it and sends stderr back to the agent.
"""

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_BIN = os.path.join(ROOT, ".venv", "bin", "python")
BASELINE = os.path.join(ROOT, "eval", "results", "baseline.json")
GOLD = os.path.join(ROOT, "eval", "gold.jsonl")

# Anything in the engine package can move a ranking, as can the eval adapter.
# Matched as path prefixes so this survives files being added or renamed -- an
# earlier version listed modules by bare name and broke the moment the project
# was restructured into src/.
WATCHED_PREFIXES = ("src/msgsearch/", "eval/retriever")

# A drop smaller than this is noise at the current gold-set size, not a regression.
TOLERANCE = 0.02


def sh(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def touched():
    diff = sh("git", "diff", "HEAD", "--name-only") + sh("git", "diff", "--name-only")
    files = {line.strip() for line in diff.splitlines() if line.strip()}
    return sorted(f for f in files if f.startswith(WATCHED_PREFIXES))


def labeled_queries():
    if not os.path.exists(GOLD):
        return 0
    with open(GOLD) as f:
        return sum(1 for line in f if line.strip() and json.loads(line).get("relevant"))


def already_blocked_for(state):
    """Block at most once per code state, so a change the agent cannot fix
    does not trap it in a loop."""
    marker = os.path.join(tempfile.gettempdir(), f"msgsearch-hook-{state}")
    if os.path.exists(marker):
        return True
    open(marker, "w").close()
    return False


def main():
    with contextlib.suppress(Exception):
        json.load(sys.stdin)

    files = touched()
    if not files or not os.path.exists(PY_BIN):
        return 0
    if labeled_queries() < 3 or not os.path.exists(BASELINE):
        return 0

    proc = subprocess.run(
        [PY_BIN, "eval/bench.py", "-r", "eval.retriever", "--against", "baseline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout
    match = re.search(r"^mrr\s+([\d.]+)", out, re.M)
    if not match:
        return 0

    mrr = float(match.group(1))
    base = json.load(open(BASELINE))["metrics"]["mrr"]
    if mrr >= base - TOLERANCE:
        return 0

    state = hashlib.sha1((sh("git", "diff", "HEAD") + f"{mrr}").encode()).hexdigest()[:12]
    if already_blocked_for(state):
        return 0

    print(
        f"Retrieval regression: MRR {base:.4f} -> {mrr:.4f} "
        f"({mrr - base:+.4f}) after editing {', '.join(files)}.\n\n"
        f"{out.strip()}\n\n"
        "Either fix it, or revert the change. If the drop is intended and "
        "justified, say so explicitly and re-snapshot with "
        "`eval/bench.py -r eval.retriever --save baseline`.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
