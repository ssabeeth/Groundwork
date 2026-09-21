# Working on Groundwork

Retrieval over scientific literature, where the point is the measurement, not the
pipeline. Read `docs/decisions.md` before changing anything in `eval/` or `bm25.py` —
the non-obvious choices are explained there and most of them are load-bearing.

## The one rule

A baseline exists before the method that would beat it. Every result is compared
against a number that was established first and tested independently. If you find
yourself building a retrieval method before its baseline is measured, stop.

## Tests

**Write tests from the definition, never from what the code returns.** This is the
single most important convention in the repo and it is easy to break without noticing.

- Expected values in `tests/test_metrics.py` are literals worked out by hand, with the
  arithmetic shown in comments. Do not replace them with values copied from a failing
  test's "Obtained:" line.
- `tests/test_bm25.py` checks the optimised scorer against `naive_bm25`, a separate
  transcription of the Lucene formula in the test file. Keep the two implementations
  independent. Do not refactor the naive one to call into the package.
- If a test fails, work out which side is wrong on paper before changing either.

A test that asserts the implementation agrees with itself proves nothing, and the
whole project rests on these numbers being right.

## Results

- Every number in the README comes from `scripts/run_baseline.py` and is reproducible
  with one command.
- Record settings with results. Tokenisation, `k1`, `b`, and gain function all move
  scores by as much as the effects an ablation is trying to detect. `describe()` on
  retriever and tokenizer exists for this; keep it current when adding parameters.
- Add an entry to `docs/experiments.md` for every run that settles something.
- **Failed experiments stay in the log with their numbers.** Do not tidy them away. A
  method that did not help is a result.
- Do not claim a win under about a point of nDCG on 300 queries without a paired
  significance test.

## Code

- Python 3.11, `src/` layout, type hints on all signatures.
- `ruff check .` and `ruff format .` clean before committing. `pytest` green.
- Core dependencies are numpy and tqdm only. Stemming is an optional extra. Think
  before adding a third — a heavy dependency tree is a reason people do not run
  portfolio code.
- New retrievers implement the `Retriever` protocol in `retrieval/base.py` so the eval
  harness does not need to know which method it is scoring.
- Never commit anything under `data/`.

## Changing defaults

`k1=0.9`, `b=0.4`, exponential nDCG gain and Lucene-variant IDF are chosen for
comparability with published BEIR baselines, not arbitrarily. Changing any of them
means updating `docs/decisions.md` in the same commit and re-running affected results.

## Commands

```bash
pytest
ruff check . && ruff format .
python scripts/run_baseline.py --dataset scifact
python scripts/run_baseline.py --dataset scifact --no-stem --tag no-stemming
```

## Roadmap position

Milestone 1 (tested metrics, BM25 baseline) is done. Next is the tokenisation
ablation, then dense retrieval on the same harness. See the pending table in
`docs/experiments.md`.
