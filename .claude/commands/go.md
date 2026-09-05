---
description: Verify, simplify, and commit the current change
---
Finish the current change end to end:

1. **Verify** — run `python3 eval/bench.py --against baseline` and paste the table.
   For non-retrieval changes, run the affected script directly and show real output.
   Never claim something works without having run it.
2. **Simplify** — reread the diff. Remove dead code, collapse needless abstraction,
   and check it matches the conventions in AGENTS.md.
3. **Record** — if you got anything wrong during this task, append the correction to
   the "Learned corrections" section of AGENTS.md. This is not optional; it is how
   the project gets easier over time.
4. **Commit** — stage and commit with a message describing the *why*. Do not push
   unless asked.

Confirm no message bodies appear in the diff before committing.
