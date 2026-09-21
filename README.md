# Groundwork

Retrieval over scientific literature, with the numbers measured rather than assumed.

Most RAG projects pick a retrieval method and move on. This one treats the choice as
the experiment: BM25, dense, hybrid and reranked retrieval are scored against the same
human relevance judgements, with a tested evaluation harness, and the results are
reported including the ones that did not help.

**Status:** milestone 1 — BM25 baseline with a tested nDCG implementation. No dense
retrieval, no hybrid, no reranking yet, by design. Everything later is compared to
this number, so the number has to be trustworthy first.

---

## The question

When you retrieve over scientific text, how much does retrieval method actually
matter, and where? Semantic search is the default recommendation, but scientific
queries are full of exact tokens — gene symbols, compound names, species binomials,
accession numbers — where lexical matching is hard to beat. The claim this project
tests is that the answer is query-dependent, and that the breakdown by query type is
more useful than a single headline score.

## The data

Evaluation uses [BEIR](https://github.com/beir-cellar/beir), which ships queries with
human relevance judgements, so the metrics are comparable to published work:

| Dataset | Documents | Test queries | Judgements | Domain |
|---|---|---|---|---|
| SciFact | 5,183 | 300 | binary | Scientific claim verification |
| TREC-COVID | 171,332 | 50 | graded 0–2 | COVID-19 literature |
| NFCorpus | 3,633 | 323 | graded 0–2 | Nutrition and medical |

SciFact is first because it is small enough to iterate on in seconds. It is also the
dataset where BM25 is already strong, so it is a good harness test and a poor showcase
for hybrid retrieval — TREC-COVID is where an ablation has room to show something.

Datasets download on first use into `data/`, which is gitignored.

## The method

**BM25, Lucene variant.** Written out in
[`src/groundwork/retrieval/bm25.py`](src/groundwork/retrieval/bm25.py) rather than
imported, for two reasons. First, the baseline every later result is measured against
should be understood, not trusted. Second, the common pip BM25 packages use the
Robertson IDF, which goes negative for terms appearing in more than half the corpus;
Lucene's form cannot, and the published BEIR baselines are Lucene's.

```
idf(q) = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
score  = Σ_q idf(q) · tf(q,D)(k₁+1) / (tf(q,D) + k₁(1 - b + b·|D|/avgdl))
```

Defaults `k1=0.9, b=0.4` — the BEIR paper's settings, not Lucene's out-of-the-box
`k1=1.2, b=0.75`. Results across those two settings are not comparable, so the values
used are recorded with every run.

**Metrics.** nDCG@k and recall@k in
[`src/groundwork/eval/metrics.py`](src/groundwork/eval/metrics.py), following
`trec_eval` conventions, with 21 unit tests against values worked out by hand. Three
decisions that are the usual cause of numbers that do not reproduce:

- **Gain is exponential** (`2^rel - 1`), as `trec_eval`'s `ndcg_cut` uses. For binary
  qrels like SciFact this is identical to linear gain, since `2¹ - 1 = 1`. For graded
  qrels it is not, and it weights a level-2 document three times a level-1 one rather
  than twice. Both are implemented; which one was used is recorded.
- **IDCG is built from every judged-relevant document**, not only the retrieved ones,
  so a system that misses relevant documents entirely is penalised for it.
- **Queries in the qrels but absent from the run score zero** rather than being
  dropped from the average. Dropping them inflates the mean.

**Tokenisation is explicit.** Stemming alone moves BEIR nDCG@10 by a point or two, so
stopwords and stemming are configurable, off the Lucene English stopword list by
default, and written into the results file with every run.

## Results

Reproduce with one command:

```bash
python scripts/run_baseline.py --dataset scifact
```

| Dataset | Method | nDCG@10 | Recall@100 | BEIR published BM25 nDCG@10 |
|---|---|---|---|---|
| SciFact | BM25 (k1=0.9, b=0.4, Porter) | _pending first run_ | _pending_ | 0.665 |

The script prints the delta against the published figure and exits non-zero if it is
more than 0.03 away. That tolerance is deliberate: a reimplementation will not match
Elasticsearch to three decimals, because tokenisation and stemming differ, but landing
well outside it means the harness is wrong rather than the method. This is the main
reason the BM25 baseline exists before anything else — it is a correctness check on
the evaluation code that everything downstream depends on.

Full run records, including settings and timings, are written to `results/`.

## What didn't work

Nothing to report yet — this section fills in as experiments run, and failed ones stay
here with their numbers.

## Limitations

- Only SciFact is wired up so far; TREC-COVID and NFCorpus are supported by the loader
  but not yet run.
- `trec_eval` ties are broken by document id here. `pytrec_eval` breaks them
  differently, so runs with many exactly-tied scores can differ in the fourth decimal.
- Recall@100 caps at the retrieval depth; deeper retrieval would change it.
- No significance testing yet. Differences under roughly a point on 300 queries should
  not be read as real until paired tests are added.

## Roadmap

1. ~~Tested metrics and BM25 baseline~~ ← current
2. Dense retrieval, and the same numbers on the same harness
3. Hybrid (reciprocal rank fusion), plus a cross-encoder reranker
4. Results broken down by query type — the actual question
5. Answer generation with citations, LLM judge calibrated against human labels
6. MCP server so it plugs into any assistant

## Install

```bash
conda env create -f environment.yml
conda activate groundwork
```

or

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,stem]"
```

## Development

```bash
pytest          # 64 tests
ruff check .
ruff format .
```

CI runs lint, format check and tests on every push.

## Layout

```
src/groundwork/
  data/beir.py          dataset download and loading
  retrieval/bm25.py     Lucene-variant BM25
  retrieval/tokenize.py tokenisation, stopwords, stemming
  eval/metrics.py       nDCG@k, recall@k
scripts/run_baseline.py the one command behind the results table
docs/experiments.md     running log, including what failed
tests/                  64 tests
```

## Licence

MIT.
