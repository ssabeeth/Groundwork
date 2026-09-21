# Groundwork

Retrieval over scientific literature, with the numbers measured rather than assumed.

Most RAG projects pick a retrieval method and move on. This one treats the choice as the
experiment: BM25, RM3, dense, hybrid and reranked retrieval are scored against the same
human relevance judgements, with a tested evaluation harness, and the results are
reported including the ones that did not help — and including the ones this project got
wrong and had to take back.

**Status:** twelve experiments across four BEIR datasets. Every method is measured on one
harness with paired significance testing throughout, every figure in this file is checked
against a committed run by CI, and two hypotheses were pre-registered in the log before
the data was touched. Start with the note below: three results here replaced earlier
results of this project's own, and one pre-registered prediction failed outright.

---

## Note: what this project got wrong, and what changed

Read this before the numbers.

Four things below are corrections this project made to itself. They are at the top rather
than buried because a benchmark that reports only the results that survived is not
reporting a measurement, it is reporting a selection. The superseded numbers are kept, not
deleted — Annexe B says where.

**1. The BM25 baseline was not indexing the way BEIR does.** For nine experiments,
`bm25.py` concatenated title and body into one bag of words, and its docstring claimed
this matched BEIR's convention. It does not. BEIR indexes "the title (if available) and
passage as separate fields", each with its own lengths and its own document frequencies.
Nobody checked until a reproduction failure on TREC-COVID forced the question. Indexing
the way BEIR does brings all four published baselines inside tolerance, three of them to
within 0.0014; concatenation missed TREC-COVID by 0.0916. **Every BM25-derived number in this repository was re-run**, and
the entries whose conclusions moved say so explicitly — one of them reversed sign.

**2. "Dense retrieval never beats BM25 at ranking" was a claim about one weak model.**
Experiment 5 reported it. Experiment 9 re-ran the identical comparison with a better
encoder and the point estimates changed sign on both datasets. The corrected claim is
not "dense beats BM25" — neither reversal clears Holm across its family — it is that the
earlier claim is not supportable. Measuring a method with a badly chosen model and
reporting the result as a property of the method is an easy mistake to make and an
easier one to leave standing.

**3. The central query-dependence result is about half the size first claimed.** It was
attacked deliberately, by re-running it against a stronger encoder on the reasoning that
"BM25 wins where the query has rare terms" and "BM25 wins where the weak model fails"
are otherwise indistinguishable. It half-broke: consistent in direction across every
dataset and encoder tested, but which dataset clears p < 0.05 depended on which encoder
measured it. Experiment 11 added a third dataset with 1,000 queries — pre-registered,
committed before the data was touched — to settle the size.

**4. For a while, the published repository did not run.** A `data/` line in `.gitignore`
matched `src/groundwork/data/` at any depth, so the BEIR loader was never committed. The
README's one-command reproduction failed for anyone cloning it, and CI passing locally
hid it. Found by cloning the published repo and running it, which is now the only claim
about reproducibility this project is willing to make on its own say-so.

---

## The question

When you retrieve over scientific text, how much does retrieval method actually matter,
and where? Semantic search is the default recommendation, but scientific queries are full
of exact tokens — gene symbols, compound names, species binomials, accession numbers —
where lexical matching is hard to beat. The claim this project tests is that the answer is
query-dependent, and that the breakdown by query type is more useful than a single
headline score.

It was measured, attacked with a stronger encoder, and then pre-registered on a third
dataset chosen because it was hostile to it. **It did not replicate.** What survives is a
narrower claim about where lexical matching is a mechanism of relevance at all — and the
answer to the original question turns out to be less useful than the question assumed. That
is in the results below rather than in a footnote, because a benchmark that quietly drops
its own failed prediction is not measuring anything.

## The data

Evaluation uses [BEIR](https://github.com/beir-cellar/beir), which ships queries with
human relevance judgements, so the metrics are comparable to published work:

| Dataset | Documents | Test queries | Judgements | Train split | Domain |
|---|---|---|---|---|---|
| SciFact | 5,183 | 300 | binary | yes | Scientific claim verification |
| TREC-COVID | 171,332 | 50 | graded 0–2 | no | COVID-19 literature |
| NFCorpus | 3,633 | 323 | graded 0–2 | yes | Nutrition and medical |
| SciDocs | 25,657 | 1,000 | binary | no | Citation prediction between papers |

SciFact is first because it is small enough to iterate on in seconds. It is also the
dataset where BM25 is already strong, so it is a good harness test and a poor showcase for
hybrid retrieval.

SciDocs was added last and deliberately: it has more test queries than SciFact and
NFCorpus combined, which is what it takes to pin down an effect the size those two
suggested, and it is *hostile* to this project's central hypothesis. Its queries are paper
titles and relevance means "this paper cites that one", which is a semantic relation
rather than a lexical one — so it is the dataset where a claim about BM25's per-query
advantage is most likely to break. It has no train split, which constrains what may be
tuned on it; see `docs/decisions.md`.

Datasets download on first use into `data/`, which is gitignored.

## The method

**BM25, Lucene variant.** Written out in
[`src/groundwork/retrieval/bm25.py`](src/groundwork/retrieval/bm25.py) rather than
imported, for two reasons. First, the baseline every later result is measured against
should be understood, not trusted. Second, the common pip BM25 packages use the Robertson
IDF, which goes negative for terms appearing in more than half the corpus; Lucene's form
cannot, and the published BEIR baselines are Lucene's.

```
idf(q) = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
score  = Σ_q idf(q) · tf(q,D)(k₁+1) / (tf(q,D) + k₁(1 - b + b·|D|/avgdl))
```

**Title and body are separate fields.** Each field carries its own document lengths, its
own average length and its own document frequencies; a query scores against each
independently and the scores are summed. This is what BEIR's baselines do, and for nine
experiments this repository did not — see the note above. `--single-field` restores
concatenation on every script that builds an index.

Defaults `k1=0.9, b=0.4` — the BEIR paper's settings, not Lucene's out-of-the-box
`k1=1.2, b=0.75`. Results across those two settings are not comparable, so the values used
are recorded with every run.

**Metrics.** nDCG@k and recall@k in
[`src/groundwork/eval/metrics.py`](src/groundwork/eval/metrics.py), following `trec_eval`
conventions, with unit tests against values worked out by hand. Three decisions that are
the usual cause of numbers that do not reproduce:

- **Gain is exponential** (`2^rel - 1`), as `trec_eval`'s `ndcg_cut` uses. For binary
  qrels like SciFact and SciDocs this is identical to linear gain, since `2¹ - 1 = 1` —
  verified, all six metrics bit-identical. For graded qrels it is not, and the difference
  is measured: +0.0008 on NFCorpus but **−0.0286** on TREC-COVID. The sign is not
  predictable, because exponential gain rewards separating grade 2 from grade 1 and
  penalises a ranking blind to the distinction. Both gains are computed from the same
  retrieval pass and both are recorded.
- **IDCG is built from every judged-relevant document**, not only the retrieved ones, so a
  system that misses relevant documents entirely is penalised for it.
- **Queries in the qrels but absent from the run score zero** rather than being dropped
  from the average. Dropping them inflates the mean.

**Tokenisation is explicit.** Stemming moves BEIR nDCG@10 by around three points, so
stopwords and stemming are configurable, off the Lucene English stopword list by default,
and written into the results file with every run.

**Parameters are tuned on train and scored once on test.** `k1`, `b`, RM3's feedback
settings, the fusion constant and the routing threshold are all chosen on a split they are
not reported on. Experiment 3 exists largely to show what that discipline costs: a sweep
that looked like a win on the split that chose it was worth nothing on held-out data.

**Differences are tested, not eyeballed.** Two runs are compared with a paired
randomisation test in
[`src/groundwork/eval/significance.py`](src/groundwork/eval/significance.py), two-sided,
with Holm-Bonferroni across each family of comparisons. Per-query scores are committed
alongside every result so any comparison can be reproduced from a clean clone. The
implementation is tested against exhaustive enumeration of all `2^n` sign assignments on
small inputs — the same independent-reference trick used for BM25 — since a sampled test
checked only against itself proves nothing.

**The documentation is tested.** `tests/test_documentation.py` extracts every three- and
four-decimal figure from this README and from `docs/`, and fails the build unless each one
traces to a value in a committed results file. It also checks that a results filename
agrees with how its index was actually built. Both checks exist because the failure they
catch had already happened.

## What the measurements show

Reproduce any of it with one command:

```bash
python scripts/run_baseline.py --dataset scifact
```

Full tables are in **Annexe A**. The findings, in the order they are worth knowing:

**The harness reproduces published baselines, which is the only reason to trust anything
below it.** Four BEIR datasets, four published BM25 figures, all inside tolerance and
three of them within 0.0014 — SciFact −0.0014, NFCorpus +0.0003, SciDocs −0.0003,
TREC-COVID −0.0198. Landing within a thousandth of numbers produced by different software
years earlier is not something a reimplementation does by coincidence. It took a
correction to get there; see the note at the top.

**Encoder choice mattered more than every other decision measured, combined.** Swapping
MiniLM-L6 for bge-small moves SciFact dense retrieval from 0.6451 to 0.7200 — a larger
gain than fusion, reranking, RM3, stemming and the entire `k1`/`b` parameter space put
together. Before reaching for an architecture, try a better encoder.

**Two methods earned their place, and each only on one dataset.** RM3 gains +0.0188
nDCG@10 on NFCorpus (Holm p 0.0006) and nothing on SciFact (−0.0086, p 0.117). Fusion
gains +0.0219 over its stronger parent on NFCorpus (Holm p 0.0010) and **+0.0007 on
SciFact (p 0.94)** — nothing at all. The same recall ceiling explains both: BM25 has
already found 90.1% of what exists on SciFact and 25.9% on NFCorpus, so on SciFact there
is almost nothing left for a second system to add.

**The cheapest thing measured is worth more than most of the expensive ones.** Porter
stemming is worth +0.0309 nDCG@10 on SciFact (Holm p 0.0156) — larger than RM3, larger
than fusion on the same dataset, and free. Stopword removal, `k1`/`b` tuning, chunking and
a domain-matched encoder are all worth nothing measurable. Before reaching for a second
retrieval system, check the tokeniser.

**One method costs the most and earns the least.** Cross-encoder reranking is by a wide
margin the most expensive thing here, and it lowered nDCG@10 in three of the four
configurations tried — the exception being a +0.0062 on NFCorpus that does not survive
testing.

**The central claim did not survive its own test.** BM25's per-query advantage rises with
the rarity of the query's rarest term on SciFact and NFCorpus. On SciDocs, pre-registered
and with 1,000 queries, it does not: rho −0.0127 and +0.0019, with terciles flat and not
even monotone. Across all six tests of the hypothesis, two survive Holm correction. What is
left is a narrower and more defensible statement — term rarity tracks BM25's relative
standing where term overlap is a mechanism of relevance, and SciDocs' citation-prediction
task is one where it is not.

**Encoder quality does not transfer between corpora.** bge-small beat MiniLM-L6 decisively
on SciFact (0.7200 against 0.6451) and lost to it on SciDocs (0.1973 against 0.2164), while
truncating far less of the input. "Use the better encoder" is sound; "bge-small is the
better encoder" is a statement about two corpora, not about the model.
## What didn't work

**Cross-encoder reranking costs the most and earns almost nothing — but not quite
nothing, and that changed.** Rescoring the top 100 fused candidates with
`ms-marco-MiniLM-L-6-v2` takes minutes of compute per dataset against seconds for the
fusion it reranks. On nDCG@10 it lowered the score in three of four configurations and
none of the four differences survive correction (Holm 0.0936 to 0.6791).

On nDCG@1 there is one real result: **+0.0588 on NFCorpus over the MiniLM fusion, Holm
p 0.0296.** Reranking helping the very top of the ranking and nothing below it is the
textbook story for cross-encoders, and this project previously reported that the story did
*not* hold — an earlier version of this README used exactly that number as its example of
a paired test separating a finding from an anecdote, at p 0.208. Under correct indexing it
is a finding. It holds in one of four cells, so it is a lead rather than a conclusion, and
the honest summary is that reranking buys you rank 1 on one dataset and nothing else
anywhere.

The likely reason for the rest is the same one behind the bi-encoder results: MS MARCO is
web-search queries against web passages, and neither scientific claims nor consumer-health
phrases resemble that. This bounds the claim — *this* reranker does not help here; a
domain-matched one is untested.

**A biomedical encoder was worse than a general one, on a medical corpus.**
S-PubMedBert-MS-MARCO scores 0.3142 on NFCorpus — below BM25 (0.3253) and below both
general encoders tested (mpnet 0.3346, bge-small 0.3391). I predicted the opposite when
proposing the experiment. Whatever it gains from biomedical pretraining it loses to
bge-small's retrieval training, and "use a domain model for a domain corpus" is not
supported by anything measured here.

**Chunking to remove truncation made things slightly worse.** Experiment 5 flagged that
71–79% of documents were truncated and it sounded damning. Eliminating it by scoring
documents on their best-matching 200-word window does not help: unchunked is ahead by
0.0043 on bge-small (p 0.2557) and 0.0120 on mpnet (p 0.0259 raw, Holm 0.0518). The
discarded tails were not carrying signal, and max-pooling gives an off-topic passage a
chance to match spuriously. The caveat was worth stating and wrong about what was
limiting performance.

**Stopword removal buys nothing.** Dropping Lucene's 33-word English list moves nDCG@10
by +0.0032 with stemming on and +0.0075 with it off, neither close to significant (Holm
0.4608 and 0.2580; 1.0000 on recall@100 for both). This is the expected result rather
than a shock — IDF already discounts terms occurring in most documents, so deleting them
by list duplicates what the scoring function does. The list stays the default for
comparability with BEIR, not because it earns its place.

**A prediction this project made and got half wrong.** Experiment 4 measured that BM25 had
already captured over 90% of achievable recall@100 on SciFact, and inferred that a better
method had almost nothing to win there whatever its quality. RM3 confirmed it. Fusion
refuted it — +0.0428 nDCG@10 over BM25 at Holm p 0.0004. A recall ceiling bounds what
*recall-limited* methods can gain and says nothing about reordering documents already
retrieved. The claim was right about RM3 for the right reason and wrong about fusion for a
reason the original argument never considered.

**The expensive method was not reliably better than the cheap one.** Fusion beats RM3 on
SciFact (+0.0514, Holm p 0.0004) but not on NFCorpus (+0.0053, Holm p 0.9501). RM3 needs
numpy, runs in seconds, and has no model to download, no GPU and no 2 GB dependency tree.
Before deploying a GPU to serve fusion, it is worth knowing that on one of two datasets a
far cheaper method was statistically indistinguishable from it.

**Tuning `k1` and `b` bought exactly nothing.** A 121-cell sweep on SciFact's 809 train
queries picked `k1=1.4, b=0.3`. Carried to held-out test it scores +0.0000 against BEIR's
`0.9/0.4` at p 0.9899 — the two runs are within a ten-thousandth of each other. The
surface says why: nDCG@10 spans 0.0143 across the whole grid and 93% of cells sit within
0.010 of the best. There is no peak here, only a cliff to stay off at very low `k1`.

This result got *more* decisive after the indexing correction, and it is also the one that
exposed a bug: the sweep script was never migrated to multi-field, so for several hours it
was tuning on one index and scoring on another. Both numbers agreed that tuning is
worthless, which is the only reason the mistake was cheap.

**Routing queries by term rarity loses to simply fusing the systems.** The project's
central finding says BM25 wins on lexically specific queries, so routing each query to
BM25 or the encoder on that basis ought to beat both. It does not: −0.0069 against always
using the encoder on SciFact, +0.0034 on NFCorpus, and behind reciprocal rank fusion on
both.

The oracle router — which picks the better system per query by reading the labels, and so
bounds every possible router — scores 0.7750, 0.3887 and 0.2313, worth 3 to 5.5 points
over the better single system on all three datasets. The information is genuinely there.
`max_idf` finds essentially none of it, and a method that uses both systems on every query
beats one that has to guess which to use. A correlation of rho +0.09 to +0.17 does not
support a decision rule, and this is what that sentence means in practice.

## Limitations

- TREC-COVID's 50 queries give it very little statistical power, so it is excluded from
  method comparisons and used only as a reproduction check.
- `trec_eval` ties are broken by document id here. `pytrec_eval` breaks them differently,
  so runs with many exactly-tied scores can differ in the fourth decimal.
- Recall@100 caps at the retrieval depth; deeper retrieval would change it.
- Significance is a paired randomisation test with Holm-Bonferroni across each family of
  comparisons; seed and resample count are recorded, because p-values near a threshold
  move in the third decimal between seeds. It answers whether a difference is
  distinguishable from noise on *this* query set, not whether it generalises or whether it
  is large enough to care about — which is why effect sizes are always reported next to it.
- SciFact's and NFCorpus' ~300 queries are not many, and per-query scores are identical
  under most configuration changes, so the effective sample behind any comparison is far
  smaller than the query count. SciDocs' 1,000 queries were added for exactly this reason.
- The reference tolerance against published baselines is absolute (±0.03), so it is a much
  weaker check on a dataset scoring 0.158 than on one scoring 0.665. This was written down
  before SciDocs was run rather than after seeing which way it went.
- Every dense and reranking result uses models trained on web search text. A
  domain-matched bi-encoder was tried and was worse; a domain-matched *cross-encoder* is
  untested, so "reranking does not help here" is a claim about `ms-marco-MiniLM-L-6-v2`.

## Roadmap

Numbered as in [`docs/experiments.md`](docs/experiments.md), which is the full log.

1. ~~Tested metrics and BM25 baseline~~
2. ~~Tokenisation ablation, with paired significance testing~~
3. ~~`k1`/`b` sweep, tuned on train and checked on held-out test~~
4. ~~BM25 on graded qrels: NFCorpus and TREC-COVID~~
4b. ~~RM3 pseudo-relevance feedback~~
5. ~~Dense retrieval on the same harness~~
6. ~~Hybrid by reciprocal rank fusion~~
7. ~~Cross-encoder reranking over the fused candidates~~
8. ~~Results broken down by query type — the actual question~~
9. ~~Encoder choice and chunking, as a falsification test of 8~~
10. ~~Multi-field indexing, and the migration that followed~~
11. ~~SciDocs as a third dataset, hypotheses pre-registered~~
12. ~~Query routing: is the central finding actionable?~~

Still planned:

13. Answer generation with citations, LLM judge calibrated against human labels
14. MCP server so it plugs into any assistant

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

Dense and hybrid retrieval need the optional `dense` extra, which pulls in torch:

```bash
pip install -e ".[dense]"
```

The MCP server needs its own extra:

```bash
pip install -e ".[dense,mcp]"
python -m groundwork.server --dataset scifact
```

The core package stays on numpy and tqdm; everything except dense retrieval, generation
and the server runs without it. The `dense` extra is pinned rather than ranged, because the obvious unpinned
specification resolves to a torch/numpy combination that fails at import — see
`docs/decisions.md`.

## Development

```bash
pytest          # 394 tests
ruff check .
ruff format .
```

CI runs lint, format check and tests on every push.

## Layout

```
src/groundwork/
  data/beir.py          dataset download, loading, published baselines
  retrieval/bm25.py     Lucene-variant BM25, single- and multi-field
  retrieval/rm3.py      RM3 pseudo-relevance feedback
  retrieval/dense.py    bi-encoder dense retrieval (optional extra)
  retrieval/fusion.py   reciprocal rank fusion
  retrieval/rerank.py   cross-encoder reranking
  retrieval/routing.py  per-query system selection, and its oracle ceiling
  retrieval/expansion.py  query and document expansion (HyDE, doc2query)
  retrieval/tokenize.py tokenisation, stopwords, stemming
  eval/metrics.py       nDCG@k, recall@k, per-query scoring, oracle recall
  eval/significance.py  paired randomisation test, Spearman, Holm-Bonferroni
  eval/agreement.py     Cohen's kappa, for scoring a judge against a human
  eval/judge.py         an LLM relevance judge, and what it is worth
  generate.py           answers with citations, checked against what was retrieved
  server.py             MCP server (optional extra)
scripts/run_baseline.py  the one command behind the results table
scripts/run_sweep.py     k1/b grid over one shared index
scripts/run_rm3.py       RM3, with its own train-tuned sweep
scripts/run_dense.py     bi-encoder retrieval, embeddings cached
scripts/run_hybrid.py    RRF over BM25/dense/RM3, k tuned on train
scripts/run_router.py    per-query routing against its oracle ceiling
scripts/run_expanded.py  BM25 over expanded queries or documents
scripts/run_judge.py     an LLM judge scored against human assessors
scripts/generate_expansions.py     generate expansions once, and commit them
scripts/compare_runs.py            paired significance test between two runs
scripts/analyse_queries.py         the pre-specified query-type hypothesis
scripts/combine_query_analyses.py  Holm correction across every test of it
scripts/diagnose_query_fields.py   TREC-COVID query formulation spread
docs/experiments.md     running log, including what failed
docs/decisions.md       why the load-bearing choices are what they are
tests/                  394 tests
```

## Annexe A — full results

Every figure below is read out of a JSON file in `results/`, written by a script in this
repository, and checked by `tests/test_documentation.py`: a number in this README that no
run produced fails the build. BM25 indexes title and body as separate fields, which is
what BEIR does; `--single-field` reproduces the pre-migration behaviour.

### A.1 BM25 against the published baselines

```bash
python scripts/run_baseline.py --dataset scifact
```

| Dataset | nDCG@10 | Recall@100 | BEIR published | Delta |
|---|---|---|---|---|
| SciFact | 0.6636 | 0.9009 | 0.665 | **−0.0014** |
| NFCorpus | 0.3253 | 0.2494 | 0.325 | **+0.0003** |
| SciDocs | 0.1577 | 0.3567 | 0.158 | **−0.0003** |
| TREC-COVID | 0.6362 | 0.1139 | 0.656 | −0.0198 |

The script prints the delta and exits non-zero beyond ±0.03. That tolerance is absolute,
so it is a far weaker check at 0.158 than at 0.665 — stated here because it was written
into the SciDocs pre-registration before the number was known, not after.

### A.2 Every method on the same harness

nDCG@10 on test, with recall@100 in brackets:

| Method | SciFact | NFCorpus | SciDocs |
|---|---|---|---|
| BM25 | 0.6636 (0.9009) | 0.3253 (0.2494) | 0.1577 (0.3567) |
| RM3 | 0.6550 (0.9053) | **0.3440** (0.3121) | — |
| Dense — MiniLM-L6 | 0.6451 (0.9250) | 0.3173 (0.3115) | **0.2164** (0.5101) |
| Dense — mpnet | — | 0.3346 | — |
| Dense — S-PubMedBert | — | 0.3142 | — |
| Dense — bge-small | **0.7200** (0.9533) | 0.3391 (0.3059) | 0.1973 (0.4627) |
| Hybrid RRF (BM25+MiniLM) | 0.7064 (0.9550) | 0.3493 (0.3217) | 0.2012 (0.4804) † |
| Hybrid RRF (BM25+bge) | **0.7207** (0.9617) | **0.3610** (0.3135) | 0.1958 (0.4541) † |

† SciDocs has no train split, so its fusion constant is fixed at the conventional `k=60`
in advance rather than swept. The tuned `k` on the other two datasets came out between 1
and 10, so this understates what fusion is worth there. Sweeping `k` on the only split
SciDocs has would have produced a better number and no information — see
`docs/decisions.md`.

### A.3 Where the headroom is

Recall@100 is not comparable across datasets — they differ in how many relevant documents
exist per query. Against the best any system could reach given the judgements:

| Dataset | Recall@100 | Oracle | Share of ceiling |
|---|---|---|---|
| SciFact | 0.9009 | 1.0000 | 90.1% |
| TREC-COVID | 0.1139 | 0.2674 | 42.6% |
| SciDocs | 0.3567 | 1.0000 | 35.7% |
| NFCorpus | 0.2494 | 0.9647 | 25.9% |

BM25 has already found 90% of what exists on SciFact, which is why RM3 and fusion have so
little to add there and so much on NFCorpus. It bounds *recall-limited* methods only: it
says nothing about reordering documents already retrieved, which is how fusion beat it.

### A.4 Tokenisation ablation

SciFact, same corpus and parameters, nDCG@10 with recall@100 in brackets:

| | stopwords removed | stopwords kept |
|---|---|---|
| **Porter stem** | **0.6636** (0.9009) | 0.6604 (0.8987) |
| **no stem** | 0.6327 (0.8926) | 0.6252 (0.8876) |

```bash
python scripts/compare_runs.py \
  --pair results/scifact-bm25.json results/scifact-bm25-no-stem.json
```

Stemming is worth +0.0309 nDCG@10 (Holm p 0.0156) and +0.0083 recall@100 (Holm p 1.0000).
Before the indexing correction those two findings were the other way round; see the note
at the top.

### A.5 Parameter sweeps, all tuned on train and scored once on test

| Sweep | Grid | Best on train | On held-out test |
|---|---|---|---|
| BM25 `k1`/`b`, SciFact | 121 cells | `k1=1.4, b=0.3` | **+0.0000, p 0.9899** |
| RM3, SciFact | 60 cells | `fb=20, terms=50, α=0.3` | −0.0086, p 0.1168 |
| RM3, NFCorpus | 60 cells | `fb=5, terms=50, α=0.8` | **+0.0188, Holm p 0.0006** |
| RRF `k`, SciFact | 9 values | `k=1` (bge), `k=5` (MiniLM) | see A.2 |
| RRF `k`, NFCorpus | 9 values | `k=5` (bge), `k=10` (MiniLM) | see A.2 |

The `k1`/`b` surface has no peak to find. Across 121 cells nDCG@10 spans 0.0143 and 93%
of cells sit within 0.010 of the best, so the defaults are already on the plateau. The
tuned setting carried to test is worth +0.0000 — not rounded down; the two runs score
within a ten-thousandth of each other.

The fusion constant `k` lands between 1 and 10 on every dataset with a train split, far
below the conventional default of 60. That default costs real score here, which is why
SciDocs' untuned fusion in A.2 is a lower bound rather than a measurement.

### A.6 Query-dependence, corrected across every test of it

One hypothesis, asked six times. Correcting across the family is the honest treatment,
because asking one question of six query sets and reporting the smallest p-value is the
error that correction exists to prevent:

| Dataset | Encoder | Spearman rho | p | Holm |
|---|---|---|---|---|
| SciFact | bge-small | +0.1685 | 0.0043 | **0.0255** |
| NFCorpus | MiniLM-L6 | +0.1601 | 0.0042 | **0.0255** |
| SciFact | MiniLM-L6 | +0.1372 | 0.0168 | 0.0674 |
| NFCorpus | bge-small | +0.0934 | 0.0906 | 0.2719 |
| SciDocs | bge-small | +0.0019 | 0.9546 | 1.0000 |
| SciDocs | MiniLM-L6 | −0.0127 | 0.6810 | 1.0000 |

```bash
python scripts/combine_query_analyses.py \
  --analysis results/scifact-query-analysis.json \
  --analysis results/scidocs-query-analysis.json
```

Two of six survive. The SciDocs rows are not underpowered — they are the largest query
sets here, and their terciles are flat and non-monotone rather than weakly ordered.


### A.7 Query routing and its ceiling

An oracle router picks the better system per query *by reading the labels*. It is not a
method; it bounds every possible router, including ones nobody has built:

| Dataset | BM25 | bge-small | Oracle | `max_idf` router | Router vs better single |
|---|---|---|---|---|---|
| SciFact | 0.6636 | 0.7200 | **0.7750** | 0.7131 | **−0.0069** |
| NFCorpus | 0.3253 | 0.3391 | **0.3887** | 0.3425 | +0.0034 |
| SciDocs | 0.1577 | 0.1973 | **0.2313** | — | — |

```bash
python scripts/run_router.py --dataset scifact \
  --lexical results/scifact-bm25.json --semantic results/scifact-dense-bge.json \
  --lexical-train results/scifact-bm25-train.json \
  --semantic-train results/scifact-dense-bge-train.json --tag bge
```

Perfect routing is worth 3 to 5.5 points on all three datasets — more than fusion, RM3,
reranking or stemming. `max_idf` recovers about 7% of it on NFCorpus and goes backwards on
SciFact, and fusion beats the router on both. SciDocs has no train split, so only the
ceiling is reported; it needs no tuning, which is a second reason the oracle is the useful
number.

### A.8 TREC-COVID query formulation

The one BEIR dataset shipping several query forms. Indexing once and re-retrieving per
field, against the published 0.656:

| Query field | nDCG@10 | vs published | recall@100 |
|---|---|---|---|
| `text` (BEIR canonical, what the loader uses) | 0.6362 | −0.0198 | 0.1139 |
| `metadata.query` (keyword form) | 0.5841 | −0.0719 | 0.1117 |
| `metadata.narrative` | 0.4944 | −0.1616 | 0.0869 |
| `query` + `text` | 0.7107 | +0.0547 | 0.1372 |
| `query` + `text` + `narrative` | 0.7349 | +0.0789 | 0.1367 |

Formulation moves nDCG@10 by 0.2405 between the worst and best form — more than
tokenisation, `k1`/`b` and the gain function combined. It is reported here and nowhere
else because every other BEIR dataset ships one query form, so the variable is invisible
until it is not.

Experiment 4 blamed this spread for TREC-COVID's reproduction failure and was wrong; the
cause was indexing. Under concatenation, `query` + `text` landed within 0.006 of the
published figure, which would have looked like a fix and closed the investigation. With
the index corrected, the canonical field alone lands at −0.0198 and the combined forms
*overshoot*. The right number by the wrong route would have been worse than the honest
failure.

## Annexe B — superseded results, and why they are still here

Nothing in `results/` is deleted when a re-run replaces it. Three kinds of file live
there, and the distinction matters when reading the experiment log:

| Suffix | What it is |
|---|---|
| *(none)* | the current run: BM25 indexes title and body as separate fields |
| `-singlefield` | a preserved pre-migration run, title and body concatenated |
| `-multifield` | experiment 10's side-by-side comparison, kept under its own name |

Experiments 1 through 9 ran before the indexing correction, so the log entries describing
them quote the `-singlefield` numbers, and each of those entries carries a banner saying
so. They are left as they were written. Rewriting them to show the new numbers would
produce a log in which this project had never been wrong, which is the opposite of what it
is for.

What the migration changed, for the record:

| Dataset | Concatenated | Multi-field | Published |
|---|---|---|---|
| SciFact | 0.6802 (+0.0152) | 0.6636 (−0.0014) | 0.665 |
| NFCorpus | 0.3224 (−0.0026) | 0.3253 (+0.0003) | 0.325 |
| TREC-COVID | 0.5644 (−0.0916) | 0.6362 (−0.0198) | 0.656 |

Two safeguards exist because this went wrong before they did:

- **A filename asserts how its index was built**, and a test reads the retriever's own
  `describe()` to enforce it. During the migration several files briefly held concatenated
  numbers under names that had come to mean multi-field, and a log entry quoted them as
  though they were new. A reader caught it; nothing else did.
- **Every file that scores anything must record what produced it.** Sweep files did not,
  which is how `run_sweep.py` went through the entire migration still sweeping a
  concatenated index while every other script had moved on — invisible to both checks,
  because a record that says nothing cannot contradict itself. Preserved archives are held
  to a different requirement instead: they must carry a note saying what they are, since
  back-filling a description onto them would mean writing down settings nothing observed.

## Licence

MIT.
