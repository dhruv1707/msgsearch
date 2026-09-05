---
description: Run the retrieval eval and report the before/after metrics table
---
Run the verification loop for this project:

1. `python3 eval/bench.py --against baseline`
2. If `index.sqlite` is missing, rebuild it first with `python3 search/index.py`.
3. Report the metrics table verbatim. Do NOT summarize it away or round the numbers.
4. If any metric regressed, say so plainly and explain which change caused it.
5. If there are unlabeled gold queries, list them — they measure nothing until labeled.

A retrieval change that does not move these numbers is not an improvement.
