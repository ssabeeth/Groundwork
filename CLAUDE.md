# Groundwork — project conventions

## What this repo is

Groundwork is a scientific literature retrieval assistant with a measured evaluation of
retrieval strategies. The point of the project is the measurement, not the pipeline:
every claim in the README must be backed by a number produced by code in this repo.

Read `docs/decisions.md` before changing anything in `eval/` or `retrieval/bm25.py`.
The non-obvious choices are explained there and most of them are load-bearing.

## Non-negotiables

- A baseline exists before the method that would beat it. Every result is compared
  against a number established first and tested independently. If you find yourself
  building a retrieval method before its baseline is measured, stop.
- Every result is reproducible from a single command. No numbers that only exist in a
  notebook someone ran once.
- Every retrieval change is evaluated against the same benchmark before it is kept.
- Failed experiments stay documented. "Tried X, it was worse, here is the number" is
  the most valuable content in this project.
- No fabricated or estimated figures anywhere, including in docs and docstrings. A
  number that has not been produced by a run is marked pending, not guessed.

## Stack

- Python 3.11+
- Environment: conda (Anaconda), with `environment.yml` committed
- Testing: pytest
- Typing: type hints on all function signatures; no `Any` without a comment explaining why
- Formatting and linting: ruff (format + check), run before every commit
- Core dependencies are numpy and tqdm only; stemming is an optional extra. Think before
  adding a third — a heavy dependency tree is a reason people do not run portfolio code.

## Layout

```
src/groundwork/        library code, importable, no scripts
  data/               dataset download, parsing and loading
  retrieval/          indexing and search strategies
  eval/               metrics and benchmark runners
scripts/              thin CLI entry points, argparse, no logic
tests/                mirrors src/ module names; integration tests at top level
data/                 gitignored; never commit datasets
results/              committed; metrics as JSON, one file per experiment
docs/                 design notes and decision records
```

## Code style

- Functions do one thing. If a docstring needs "and", split the function.
- Classes only where there is state to hold. Prefer plain functions otherwise.
- Logging via the stdlib `logging` module, never bare `print` in library code. Scripts
  under `scripts/` print to stdout; that is what they are for.
- Errors fail loudly. No bare `except:`, no silently returning None on failure.
- No magic numbers scattered through modules. Parameters that affect results are
  explicit arguments with documented defaults, and are recorded with every run.

## Testing

Unit tests for every public function in `src/`. Tests run offline — no network calls,
fixtures for any external data. CI runs `ruff check`, `ruff format --check` and
`pytest` on every push.

**Write tests from the definition, never from what the code returns.** This is the
single most important convention in the repo and the easiest to break without noticing.
An untested nDCG implementation is worse than no nDCG, and a test that asserts the
implementation agrees with itself is the same thing wearing a disguise.

- Expected values in `tests/test_metrics.py` are literals worked out by hand, with the
  arithmetic shown in comments. Do not replace them with values copied from a failing
  test's "Obtained:" line.
- `tests/test_bm25.py` checks the optimised scorer against `naive_bm25`, a separate
  transcription of the Lucene formula living in the test file. Keep the two
  implementations independent; do not refactor the naive one to call into the package.
- If a test fails, work out which side is wrong on paper before changing either.

The whole project rests on these numbers being right.

## Results and the experiment log

- Every number in the README comes from `scripts/run_baseline.py` and is reproducible
  with one command.
- Record settings with results. Tokenisation, `k1`, `b` and gain function each move
  scores by as much as the effects an ablation is trying to detect. `describe()` on the
  retriever and tokenizer exists for this; keep it current when adding parameters.
- `docs/experiments.md` is the full chronological log — one entry per run that settles
  something, including the ones that failed, with their numbers. Do not tidy them away.
- The README's "what didn't work" section is the short public version of that log.
- Do not claim a win under about a point of nDCG on 300 queries without a paired
  significance test.

## Changing defaults

`k1=0.9`, `b=0.4`, exponential nDCG gain and the Lucene-variant IDF are chosen for
comparability with published BEIR baselines, not arbitrarily. Changing any of them
means updating `docs/decisions.md` in the same commit and re-running affected results.

## Deliberately not here yet

Both of these are planned, and their absence is a decision rather than an oversight.
Add them at the point named, not before:

- **MLflow, local file backend.** Comes in at experiment 5 (dense retrieval), when there
  are enough runs and parameters for JSON files in `results/` to stop being adequate.
- **YAML-loaded typed config.** Premature while parameters fit in argparse flags.
  Revisit when a run needs more than about eight.

## Git

- Small commits, present-tense messages describing the change, not the file.
- Branch per experiment when the change is big enough to revert whole.

## Commands

```bash
pytest
ruff check . && ruff format .
python scripts/run_baseline.py --dataset scifact
python scripts/run_baseline.py --dataset scifact --no-stem --tag no-stemming
```

## Where the project is

Milestone 1 (tested metrics, BM25 baseline) is done and the SciFact baseline is
measured: nDCG@10 0.6802, +0.0152 from BEIR's published 0.665 and inside the +/-0.03
tolerance, so the harness reproduces a published number. Next is the tokenisation
ablation (experiment 2), then dense retrieval on the same harness. See the pending
table in `docs/experiments.md`.

## How to work with me on this

- Propose the approach before writing code for anything non-trivial. I want to
  understand the decision, not just receive the result.
- When there is a choice between approaches, state the trade-off and recommend one.
- Explain anything I have not used before rather than assuming familiarity.
- Write complete files rather than fragments I have to splice together.
- If something I asked for is a bad idea, say so and say why.
