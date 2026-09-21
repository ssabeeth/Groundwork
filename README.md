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
  qrels like SciFact this is identical to linear gain, since `2¹ - 1 = 1` — verified,
  all six metrics bit-identical. For graded qrels it is not, and the difference is
  measured: +0.0014 on NFCorpus but **−0.0253** on TREC-COVID. The sign is not
  predictable, because exponential gain rewards separating grade 2 from grade 1 and
  penalises a ranking that is blind to the distinction. Both gains are computed from
  the same retrieval pass and both are recorded.
- **IDCG is built from every judged-relevant document**, not only the retrieved ones,
  so a system that misses relevant documents entirely is penalised for it.
- **Queries in the qrels but absent from the run score zero** rather than being
  dropped from the average. Dropping them inflates the mean.

**Tokenisation is explicit.** Stemming alone moves BEIR nDCG@10 by a point or two, so
stopwords and stemming are configurable, off the Lucene English stopword list by
default, and written into the results file with every run.

**Differences are tested, not eyeballed.** Two runs are compared with a paired
randomisation test in
[`src/groundwork/eval/significance.py`](src/groundwork/eval/significance.py), two-sided,
with Holm-Bonferroni across each family of comparisons. Per-query scores are committed
alongside every result so any comparison can be reproduced from a clean clone. The
implementation is tested against exhaustive enumeration of all `2^n` sign assignments on
small inputs — the same independent-reference trick used for BM25 — since a sampled test
checked only against itself proves nothing.

## Results

Reproduce with one command:

```bash
python scripts/run_baseline.py --dataset scifact
```

| Dataset | Method | nDCG@10 | Recall@100 | BEIR published BM25 nDCG@10 |
|---|---|---|---|---|
| SciFact | BM25 (k1=0.9, b=0.4, Porter) | **0.6802** | 0.9220 | 0.665 |
| SciFact | BM25 tuned on train (k1=1.4, b=0.5) | 0.6865 | 0.9216 | — |
| NFCorpus | BM25 (k1=0.9, b=0.4, Porter) | 0.3224 | 0.2461 | 0.325 |
| TREC-COVID | BM25 (k1=0.9, b=0.4, Porter) | 0.5644 | 0.1088 | 0.656 — **not reproduced** |
| NFCorpus | **RM3** (fb=5, terms=50, α=0.8) | **0.3433** | **0.3105** | — |
| SciFact | RM3 (fb=20, terms=20, α=0.2) | 0.6848 | 0.9253 | — |

**RM3 pseudo-relevance feedback is the first method here to beat its baseline** — and
only where there was room. On NFCorpus it gains +0.0208 nDCG@10 (Holm p = 0.0005) and
+0.0645 recall@100 (p < 0.0001). On SciFact it gains +0.0046 at p = 0.29, which is no
measurable win at all.

That split was predicted before RM3 existed, by the recall-ceiling table below: SciFact
had already captured 92% of the achievable recall@100, NFCorpus only 25%. The tuned
parameters agree from the other side — SciFact's sweep chose α = 0.2 (barely expand the
query), NFCorpus's chose α = 0.8 (largely replace it). NFCorpus queries are short
consumer-health phrases against medical writing, and closing that vocabulary gap is
exactly what feedback terms do; SciFact claims are already written in the register of
the documents they match.

Cost: retrieval goes from 0.1s to 9.7s per 300 queries — two passes plus feedback
tokenisation.

The tuned row is here to be dismissed: +0.0063 over the default on held-out test at
p = 0.217. See "what didn't work".

TREC-COVID is 0.092 below the published figure and is reported as a failed
reproduction rather than quietly fixed. It is the only BEIR dataset shipping several
query formulations, and the choice between them moves nDCG@10 by 0.24 — more than
tokenisation, `k1`/`b` and the gain function combined:

| TREC-COVID query field | nDCG@10 |
|---|---|
| `text` (BEIR canonical, used here) | 0.5644 |
| `metadata.query` | 0.5860 |
| `query` + `text` | 0.6619 |
| `query` + `text` + `narrative` | 0.6970 |

Concatenating fields would land within 0.006 of the published number. That is exactly
why it was not adopted: nothing independent says BEIR did that, and the only argument
for it is that it matches the target. `text` is what the loader uses everywhere and it
reproduces SciFact and NFCorpus without special pleading.

**Where the headroom is.** Recall@100 is not comparable across datasets — they differ in
how many relevant documents exist per query (median 1, 16 and 478). Against the best any
system could reach given the judgements:

| Dataset | Recall@100 | Oracle | Share of ceiling |
|---|---|---|---|
| SciFact | 0.9220 | 1.0000 | **92.2%** |
| NFCorpus | 0.2461 | 0.9647 | **25.5%** |
| TREC-COVID | 0.1088 | 0.2674 | **40.7%** |

BM25 has already found 92% of what exists on SciFact, so reranking and fusion have
almost nothing to win there whatever their quality. NFCorpus is where a method can show
something.

Tokenisation ablation, same corpus and parameters, nDCG@10 with recall@100 in brackets:

| | stopwords removed | stopwords kept |
|---|---|---|
| **Porter stem** | **0.6802** (0.9220) | 0.6814 (0.9197) |
| **no stem** | 0.6627 (0.8859) | 0.6611 (0.8852) |

```bash
python scripts/compare_runs.py \
  --pair results/scifact-bm25.json results/scifact-bm25-no-stem.json
```

That run is +0.0152 from the published figure, inside the tolerance — the harness
reproduces a number someone else measured with different software, which is the whole
reason this milestone exists. Landing slightly above Elasticsearch rather than below is
the expected direction for a tokenisation difference: Porter stemming plus a 33-word
stopword list normalises more aggressively than Elasticsearch's default analyzer chain.

The script prints the delta against the published figure and exits non-zero if it is
more than 0.03 away. That tolerance is deliberate: a reimplementation will not match
Elasticsearch to three decimals, because tokenisation and stemming differ, but landing
well outside it means the harness is wrong rather than the method. This is the main
reason the BM25 baseline exists before anything else — it is a correctness check on
the evaluation code that everything downstream depends on.

Full run records, including settings and timings, are written to `results/`.

## What didn't work

**Stopword removal buys nothing on SciFact.** Dropping Lucene's 33-word English list
moves nDCG@10 by about a thousandth of a point, and the sign flips depending on whether
stemming is on: −0.0012 stemmed, +0.0016 unstemmed. Neither is close to significant
(Holm-adjusted p = 1.00 on both nDCG@10 and recall@100). This is the expected result
rather than a shock — IDF already discounts terms occurring in most documents, so
deleting them by list duplicates what the scoring function does. The list stays the
default for comparability with BEIR's Elasticsearch runs, not because it earns its place.

**Tuning `k1` and `b` bought nothing.** A 121-cell sweep on SciFact's 809 train queries
picked `k1=1.4, b=0.5`, beating BEIR's `0.9/0.4` by 0.0028 nDCG@10 — on the split that
chose it. Carried to held-out test the gap is +0.0063 at p = 0.217, and −0.0004 on
recall@100. Not a win. The defaults stay.

The surface says why: across all 121 cells nDCG@10 spans 0.0268, and 74% of cells sit
within 0.010 of the best. `b` is nearly irrelevant on this corpus (0.0076 between its
best and worst, including `b=0`, which disables length normalisation entirely — SciFact
abstracts are uniform enough that there is little to normalise). `k1` only matters below
about 0.8, where it costs real score; above that the curve is flat to 2.0. There is no
peak to find here, only a cliff to stay off — and the whole parameter space is worth less
than the stemming effect that experiment 2 could not establish as significant.

Sweeping on the test split and reporting the best cell would have produced a much more
flattering number, and a meaningless one.

**Stemming's effect on ranking could not be established, only its effect on recall.**
The difference of means says stemming is worth about two points of nDCG@10 — the sort of
number that gets reported as a win. The paired test disagrees: Holm-adjusted p = 0.166
and 0.110 across the ablation's four comparisons, so on 300 queries that gap is not
distinguishable from noise. Recall@100 is a different story, a smaller-looking +0.035
that clears correction comfortably (Holm p = 0.018 and 0.021).

The per-query counts show why. On nDCG@10 stemming changes 65 queries and wins 36 of
them — close to a coin flip, with the positive mean coming from winning bigger rather
than winning more often. On recall@100 it changes only 16 queries and wins 14. The
randomisation test rewards consistency, and here consistency and magnitude point at
different metrics. Reported as a win on recall, and as undetermined on ranking.

## Limitations

- TREC-COVID's published number is not reproduced here; see the results section. Its
  50 queries also give it very little statistical power.
- `trec_eval` ties are broken by document id here. `pytrec_eval` breaks them
  differently, so runs with many exactly-tied scores can differ in the fourth decimal.
- Recall@100 caps at the retrieval depth; deeper retrieval would change it.
- Significance is a paired randomisation test with Holm-Bonferroni across each family
  of comparisons; seed and resample count are recorded, because p-values near a
  threshold move in the third decimal between seeds. It answers whether a difference is
  distinguishable from noise on *this* query set, not whether it generalises or whether
  it is large enough to care about — which is why effect sizes are always reported
  next to it.
- 300 queries is not many. SciFact's per-query nDCG@10 is identical under most
  configuration changes, so the effective sample behind any comparison is far smaller
  than 300 and the test has correspondingly little power.

## Roadmap

1. ~~Tested metrics and BM25 baseline~~
2. ~~Tokenisation ablation, with paired significance testing~~
3. ~~`k1`/`b` sweep, tuned on train and checked on held-out test~~
4. Dense retrieval, and the same numbers on the same harness
5. Hybrid (reciprocal rank fusion), plus a cross-encoder reranker
6. Results broken down by query type — the actual question
7. Answer generation with citations, LLM judge calibrated against human labels
8. MCP server so it plugs into any assistant

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
pytest          # 138 tests
ruff check .
ruff format .
```

CI runs lint, format check and tests on every push.

## Layout

```
src/groundwork/
  data/beir.py          dataset download and loading
  retrieval/bm25.py     Lucene-variant BM25
  retrieval/rm3.py      RM3 pseudo-relevance feedback
  retrieval/dense.py    bi-encoder dense retrieval (optional extra)
  retrieval/tokenize.py tokenisation, stopwords, stemming
  eval/metrics.py       nDCG@k, recall@k, per-query scoring
  eval/significance.py  paired randomisation test, Holm-Bonferroni
scripts/run_baseline.py the one command behind the results table
scripts/compare_runs.py paired significance test between two runs
scripts/run_sweep.py    k1/b grid over one shared index
scripts/run_rm3.py      RM3, with its own train-tuned sweep
docs/experiments.md     running log, including what failed
tests/                  138 tests
```

## Licence

MIT.
