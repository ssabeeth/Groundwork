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

- Every number in the README comes from a script in `scripts/` and is reproducible with
  one command. `tests/test_documentation.py` enforces this: a figure in the README or
  docs that no committed results file contains fails the build. If a number is worth
  quoting, the script that produces it writes it to `results/` — computing it in prose
  is how an unchecked figure gets in.
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

- **MLflow, local file backend.** The trigger was experiment 5, and it has been passed;
  the project is at twelve experiments and roughly seventy results files without it. The
  decision was revisited rather than left to rot: JSON files are still adequate, and they
  are now load-bearing in a way MLflow would not replace. `tests/test_documentation.py`
  reads them directly to check every documented figure, git preserves superseded runs
  under a `-singlefield` suffix, and a clean clone can reproduce any comparison from the
  committed per-query scores. Adopting MLflow now would mean either keeping both or
  rewriting that check against a database. Revisit if a run ever needs artefacts that do
  not belong in git.
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
python scripts/run_baseline.py --dataset scifact --no-stem --tag no-stem
python scripts/compare_runs.py \
    --pair results/scifact-bm25.json results/scifact-bm25-no-stem.json
```

## Where the project is

Twelve experiments, four BEIR datasets, all logged in `docs/experiments.md`. The full
numbers live in the README's annexe; what matters for working here is the shape:

**BM25 indexes title and body as separate fields, as BEIR does.** This was wrong for nine
experiments and corrected in experiment 10. `--single-field` restores the old behaviour.
Every BM25-derived result was re-run; pre-migration results are kept under a
`-singlefield` suffix with a note, because the log entries that quote them describe real
runs. Two conclusions reversed in the process and the log says which.

**The harness reproduces four published BM25 baselines**, three of them within 0.0014.
That is the reason to trust anything downstream of it.

**Two hypotheses are pre-registered** — SciDocs (experiment 11) and query routing
(experiment 12) — written into the log and committed before the data was touched. If a
prediction turns out wrong it stays on the page; that is what pre-registration is for.

**Things that are load-bearing and easy to break:**

- `tests/test_documentation.py` is the only thing standing between this repo and quiet
  documentation drift. Its traceability check works by making the haystack small, so
  adding results files weakens it — there is a guard test that fails when coverage grows
  too far. Do not "fix" that guard by raising its thresholds.
- A results filename asserts how its index was built. `-singlefield` means concatenated,
  anything else means multi-field, and a test reads the retriever's own `describe()` to
  enforce it. This exists because stale single-field files sat under current names for a
  while and a log entry quoted them as new.
- Every script that builds a BM25 index must record `retriever.describe()`. `run_sweep.py`
  did not, and so kept sweeping a concatenated index through the whole migration without
  anything noticing.

**Eighteen experiments, all complete** (roadmap numbering reaches 20; items 17 and 18
are built, not measured). Experiments 13, 14 and 15 put an LLM query expander,
doc2query and a modern reranker against the older methods with the same mechanism; seven of
those nine pre-registered predictions failed. Experiments 19 and 20 added the two methods
whose absence was the most citable gap — learned sparse and late interaction.

**SPLADE is the one modern method that wins**, and it corrected the story the others told.
Read alone, 13 and 14 say expansion does not help here. SPLADE expands too and beats both
by Holm 0.0006, so the real division is between models trained to generate plausible text
and models trained against relevance. It also killed this repository's four-times-repeated
"MS MARCO training does not transfer" line: SPLADE is MS MARCO-trained and is the best
single retriever measured here.

**Fusion still beats every single retriever**, including SPLADE — but experiment 20 found
the first fusion that does not beat its own components, which is the first crack in that
recommendation and is flagged in the log as the most interesting thing left open.

**Two things to know before adding an experiment here.** Pre-register the hypotheses in
`docs/experiments.md` and commit before touching data — three experiments have now had
predictions fail, and that is only worth anything because the prediction was on the page
first. And write hypotheses that are scoreable whatever happens: experiment 15's H3 was
conditioned on a gain existing, there were no gains, and it could not be scored at all.

**Check a reproduction before building on it, and check truncation before either.** Two
silent bugs in experiment 20 cost 0.0393 and 0.0494 nDCG@10; neither raised an error and
both produced runs that would have supported a confident wrong conclusion. A checkpoint's
own default length is a statement about its training data, not yours: ColBERT's
`doc_maxlen: 180` truncated 91% of SciFact.

**Before believing an expansion or generation step did anything**, run
`scripts/analyse_expansions.py`. It counts the new indexable terms a generated expansion
adds, costs nothing, and needs no retrieval. On SciFact it predicted experiment 13's null
result in advance: 249 of 300 expansions added no term the query did not already have.

## How to work with me on this

- Propose the approach before writing code for anything non-trivial. I want to
  understand the decision, not just receive the result.
- When there is a choice between approaches, state the trade-off and recommend one.
- Explain anything I have not used before rather than assuming familiarity.
- Write complete files rather than fragments I have to splice together.
- If something I asked for is a bad idea, say so and say why.
