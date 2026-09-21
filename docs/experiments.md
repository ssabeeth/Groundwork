# Experiment log

Chronological. Every entry records what was tried, what the number was, and what was
decided. Failed experiments stay in place with their numbers — a method that did not
help is a result, and deleting it is how a portfolio project turns into a brochure.

Entry template:

```
## YYYY-MM-DD — short title

**Question:** what this run was supposed to settle
**Setup:** dataset, method, parameters, tokenisation
**Result:** the numbers
**Read:** what it means
**Next:** what it changes
```

---

## 2026-09-21 — Evaluation harness and BM25 baseline

**Question:** Is the evaluation code correct, and does BM25 land where BEIR says it
should?

**Setup:** nDCG@k and recall@k written from the `trec_eval` definitions, 21 unit tests
against hand-computed values. Lucene-variant BM25 written out rather than imported,
tested against an independent naive implementation of the same formula across four
parameter settings and five queries. 64 tests total.

**Result:** All tests pass. Baseline run on SciFact pending.

**Read:** One hand-computed expected value in the nDCG tests was wrong in the seventh
decimal place — long division by hand, not a code bug — and the test caught it. Worth
noting because it is the whole argument for writing the tests from the definition
rather than from the implementation's output.

**Next:** Run `python scripts/run_baseline.py --dataset scifact` and record nDCG@10
against the published 0.665. If it lands outside ±0.03, the cause is almost certainly
tokenisation, in this order of likelihood: stemming on/off, stopword list, whether the
title is concatenated with the body.

---

## 2026-09-21 — BM25 baseline on SciFact

**Question:** Does the harness reproduce the BM25 number BEIR publishes for SciFact?

**Setup:** SciFact test split — 5,183 documents, 300 judged queries, 339 judgements,
all binary (level 1 only). Lucene-variant BM25, `k1=0.9`, `b=0.4`, retrieval depth 100.
Tokenisation: lowercased alphanumeric runs, Lucene's 33-word English stopword list,
Porter stemming via snowballstemmer, `min_length=1`. Exponential nDCG gain, which makes
no difference here — the qrels are binary and `2¹ - 1 = 1`. groundwork 0.1.0,
Python 3.11.16.

**Result:**

| | @1 | @10 | @100 |
|---|---|---|---|
| nDCG | 0.5533 | 0.6802 | 0.7072 |
| Recall | 0.5397 | 0.8030 | 0.9220 |

Published BEIR BM25 nDCG@10 is 0.665. This run gives 0.6802, a delta of **+0.0152**,
inside the ±0.03 tolerance. Index 15.0s, retrieve 0.1s over 300 queries. Full record
with settings and timings in `results/scifact-bm25.json`.

**Read:** The harness reproduces the published number, so the evaluation code that
every later result depends on is doing what it claims. The run lands slightly *above*
Elasticsearch's rather than below, which is the expected direction for a tokenisation
difference rather than a scoring bug: Porter stemming plus a 33-word stopword list is a
more aggressive normalisation than Elasticsearch's default analyzer chain, and on a
binary-qrel dataset that is worth about this much. Nothing needs fixing before moving on.

Two observations worth recording while they are in front of us.

Recall@1 (0.5397) sits *below* nDCG@1 (0.5533), which looks wrong at a glance and is
not. 23 of the 300 queries have more than one relevant document (distribution: 277
queries with 1, 14 with 2, 4 with 3, 3 with 4, 2 with 5). Recall@1 divides by the number
of relevant documents, so a query with 5 relevant documents scores at most 0.2 at rank 1;
nDCG@1 divides by an IDCG truncated to k=1, so the same query can score 1.0. The gap is
arithmetic, not a defect, and it will widen on the graded datasets in experiment 4.

Recall@100 of 0.9220 is a ceiling, not just a result. It is a macro-average, so the
statement it supports is that the average query is already missing 7.8% of its relevant
documents at depth 100 — not that 7.8% of all judgements are unreachable, which is a
different (micro-averaged) quantity this run does not report. Either way a reranker only
reorders the candidate set it is given, so nothing downstream of this retrieval can
exceed 0.9220 on this metric however good it is. That is the wall experiment 7 runs into, and
it is worth knowing the number before designing the reranking experiment rather than
after.

**Next:** Experiment 2, the tokenisation ablation. The +0.0152 delta is the argument for
running it: it says tokenisation is doing something measurable on this corpus, and the
ablation is what puts a number on how much.

---

## 2026-09-21 — Tokenisation ablation on SciFact

**Question:** How much of BM25's score on SciFact is tokenisation, and is either
component's contribution large enough to be distinguished from noise?

**Setup:** The full 2x2 — Porter stemming on/off crossed with Lucene's 33-word English
stopword list applied/not. SciFact test split, `k1=0.9`, `b=0.4`, depth 100, exponential
gain, everything else held fixed. Four runs, one per cell, each recorded in `results/`
with its per-query scores beside it in `results/per-query/`.

Differences are tested with a paired randomisation test (`eval/significance.py`),
100,000 sampled sign assignments, seed 0, two-sided, with Holm-Bonferroni applied across
the family of four comparisons made on this one query set. The seed matters and is
recorded: p-values near a threshold move in the third decimal between seeds, so a bare
p-value without its seed and resample count is not reproducible. Checked at seeds 0, 1
and 2; the conclusions below do not move.

**Result:**

nDCG@10 (recall@100 in brackets):

| | stopwords removed | stopwords kept |
|---|---|---|
| **Porter stem** | 0.6802 (0.9220) | 0.6814 (0.9197) |
| **no stem** | 0.6627 (0.8859) | 0.6611 (0.8852) |

The four comparisons, with Holm-adjusted p in brackets:

| Comparison | nDCG@10 | recall@100 |
|---|---|---|
| Stemming, stopwords removed | +0.0175, p 0.055 (0.166) | +0.0361, p 0.0046 (**0.018**) |
| Stemming, stopwords kept | +0.0204, p 0.028 (0.110) | +0.0344, p 0.0071 (**0.021**) |
| Stopword removal, stemmed | −0.0012, p 0.698 (1.000) | +0.0023, p 0.500 (1.000) |
| Stopword removal, unstemmed | +0.0016, p 0.514 (1.000) | +0.0007, p 1.000 (1.000) |

Stemming also costs about 40x the index time: 15-16s against 0.4s on 5,183 documents.

**Read:** Two findings, and the second one is the reason the significance machinery had
to exist before the write-up.

*Stopword removal does nothing here.* Roughly a thousandth of a point on nDCG@10, and
the sign flips depending on whether stemming is on. No comparison comes close to
significance on either metric. This is the expected result rather than a surprise — IDF
already discounts terms that appear in most documents, so deleting them by list is
mostly redundant with what the scoring function does anyway. The Lucene list stays the
default, but on the grounds given in `decisions.md` (comparability with the
Elasticsearch runs BEIR published), not because it earns anything. Recorded as a
negative result.

*Stemming helps recall, and its effect on ranking cannot be established on 300 queries.*
This is the interesting one. The difference of means says stemming is worth about two
points of nDCG@10, which is the kind of number that gets reported as a win. The paired
test says that on this query set it is not distinguishable from noise once the family of
four comparisons is accounted for (Holm 0.166 and 0.110). Meanwhile recall@100 — a
smaller-looking 3.5 points — clears correction comfortably (Holm 0.018 and 0.021).

The per-query breakdown explains the inversion, and it is worth stating because it is
exactly what a difference of means conceals. Against the unstemmed run with stopwords
removed:

| Metric | stemming wins | loses | ties |
|---|---|---|---|
| nDCG@10 | 36 | 29 | 235 |
| recall@100 | 14 | 2 | 284 |

On nDCG@10 stemming changes 65 queries and is close to a coin flip on which direction it
moves them; the positive mean comes from winning larger than it loses (+11.6 against
−6.4), not from winning more often. On recall@100 it changes only 16 queries but wins 14
of them. Consistency, not magnitude, is what the randomisation test rewards, and here
consistency and magnitude point at different metrics.

The honest summary: **stemming reliably widens the candidate pool, and any claim about
what it does to top-10 ordering is beyond what 300 queries can support.** That second
half would have been reported as a two-point win by anyone comparing means, this project
included, if the test had not been written first.

This also puts a number on a claim `decisions.md` has been making without one: stemming
moves BEIR nDCG@10 "by a point or two". It does — 1.75 to 2.04 points — and that is
simultaneously true and not significant here, which is a good illustration of why the
note was worth writing and why effect size and p-value have to be reported together.

**Next:** Experiment 3, the `k1`/`b` sweep. Two things carry forward. First, recall@100
is the metric with the statistical power on this dataset, so it deserves to be reported
alongside nDCG@10 rather than treated as secondary. Second, a sweep is a large family of
comparisons on one query set; with Holm across a grid of that size, almost nothing will
clear correction, so the sweep should be framed as choosing a setting rather than as
testing hypotheses about each cell.

---

## 2026-09-21 — k1/b sweep on SciFact

**Question:** Are BEIR's `k1=0.9, b=0.4` right for this corpus, or is the baseline
leaving something on the table?

**Setup:** 121 cells — `k1` from 0.2 to 2.0 in steps of 0.2, `b` from 0.0 to 1.0 in steps
of 0.1, with BEIR's default inserted into both axes so the setting under judgement is
actually on the grid. Porter stemming and the Lucene stopword list throughout, depth 100.

The sweep runs on the **train split** (809 queries) and the winner is then scored once on
**test** (300 queries), which is never swept. This is the whole methodological point of
the experiment. Sweeping on test and reporting the best cell does not answer the question
asked: the winner of a 121-cell grid is partly winning by luck, and its margin over the
default is inflated by the same selection that chose it. SciFact ships a train split over
the identical 5,183-document corpus, so there is no excuse for tuning on the evaluation
set. Held-out comparison uses the paired randomisation test, 100,000 resamples, seed 0.

The index is built once — 14s — and shared across all 121 cells, because `k1` and `b`
appear only in scoring and never in the postings, lengths or IDF. Whole sweep: 30.7s.
Rebuilding per cell would have taken roughly half an hour to produce identical numbers.

**Result:**

Best cell on train is `k1=1.4, b=0.5` at nDCG@10 0.6958, against the BEIR default's
0.6930. A gap of **+0.0028 on the split that chose it.**

Carried over to held-out test:

| Setting | nDCG@10 | recall@100 |
|---|---|---|
| `k1=0.9, b=0.4` (BEIR) | 0.6802 | 0.9220 |
| `k1=1.4, b=0.5` (tuned on train) | 0.6865 | 0.9216 |
| Paired difference | +0.0063, **p 0.217** | −0.0004, **p 1.000** |

270 of 300 test queries score identically under the two settings.

The surface, across all 121 train cells:

| | nDCG@10 |
|---|---|
| Best cell | 0.6958 |
| Worst cell | 0.6691 |
| **Total spread** | **0.0268** |
| Cells within 0.005 of best | 55 of 121 (45%) |
| Cells within 0.010 of best | 90 of 121 (74%) |

Best achievable at each value, maximising over the other axis:

| `b` | 0.0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.6882 | 0.6924 | 0.6949 | 0.6958 | 0.6947 | 0.6933 | 0.6927 |

| `k1` | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.4 | 1.8 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.6749 | 0.6838 | 0.6894 | 0.6940 | 0.6941 | 0.6958 | 0.6935 | 0.6911 |

**Read:** **BEIR's defaults are fine and the baseline stays as it is.** Tuning over 121
cells on 809 queries produced a setting that cannot be shown to beat the default on
held-out data — +0.0063 nDCG@10 at p 0.217, and nothing at all on recall@100. Correcting
for having asked about two metrics would only push those further from significance, so it
is not worth doing; the answer does not change.

Three things worth keeping from the surface.

*`b` barely matters on this corpus.* The best achievable nDCG@10 varies by 0.0076 across
the entire range of `b`, including `b=0`, which switches length normalisation off
completely. SciFact abstracts are of fairly uniform length, so there is little for the
normalisation to correct.

*`k1` matters only at the bottom.* Below about 0.8 it costs real score — `k1=0.2` gives up
two points — and above 0.8 the curve is flat to within a few thousandths all the way to
2.0. The BEIR default sits just inside the plateau. This is the one shape in the data:
not a peak to be found, but a cliff to stay off.

*The entire parameter space is worth less than tokenisation.* The full grid spans 0.0268
nDCG@10. Experiment 2 measured stemming at 0.0175 to 0.0204 — and could not establish
even that as significant on ranking. So a difference smaller than one already shown to be
undetectable is not a result, and a sweep is the wrong tool for finding one.

The honest conclusion is a negative one: there was nothing here to find, and the value of
running it is knowing that rather than assuming it. Keeping `0.9/0.4` also keeps
comparability with published BEIR numbers, which `decisions.md` argues is worth more than
a hundredth of a point even if a hundredth of a point had been real.

**Next:** Experiment 4, BM25 on TREC-COVID and NFCorpus. Those have graded qrels, so the
exponential-versus-linear gain choice starts to matter and must be recorded from that
entry onward. They also have very different length distributions from SciFact, which is
the natural place to check whether `b`'s irrelevance here is a property of this corpus or
of the method.

---

## 2026-09-21 — BM25 on graded qrels: NFCorpus and TREC-COVID

**Question:** Does the harness hold on graded relevance, and does the
exponential-versus-linear gain choice actually matter?

**Setup:** BM25 `k1=0.9, b=0.4`, Porter stemming, Lucene stopwords, depth 100, on
NFCorpus (3,633 documents, 323 test queries) and TREC-COVID (171,332 documents, 50 test
queries). Both have levels {1, 2}; TREC-COVID's qrels also contain 41,661 explicit
zeros and two judgements at level −1, all of which score as non-relevant.

`run_baseline.py` gained a `--gain` flag. Both gains are now computed from the *same*
retrieval pass — the ranking does not depend on the gain function, only the scoring of
it does — and both are recorded, so no graded number in this repo can be quoted without
the gain that produced it.

**Result:**

| Dataset | nDCG@10 (exp) | nDCG@10 (lin) | gain difference | published | delta |
|---|---|---|---|---|---|
| NFCorpus | 0.3224 | 0.3210 | +0.0014 | 0.325 | −0.0026 ✓ |
| TREC-COVID | 0.5644 | 0.5897 | **−0.0253** | 0.656 | **−0.0916 ✗** |

NFCorpus reproduces. TREC-COVID does not, by three times the tolerance, and the script
exits non-zero as designed.

**Read, part 1 — the gain function matters, and its sign is not predictable.**

On NFCorpus exponential gain scores *higher* than linear; on TREC-COVID it scores
0.0253 *lower*. That difference is larger than the stemming effect from experiment 2 and
larger than the entire `k1`/`b` parameter space from experiment 3.

The mechanism: exponential gain weights a level-2 document at 3 and a level-1 at 1,
against 2 and 1 for linear, so it rewards a system that separates the two grades and
penalises one that does not — the IDCG rises faster than the DCG when the ranking is
grade-blind. NFCorpus's BM25 ranking separates grades slightly; TREC-COVID's does not.
So the choice is not a convention that quietly cancels out. It is worth a quarter of the
gap being investigated below, and on binary qrels it is provably nothing at all
(verified: all six SciFact metrics are bit-identical under the two gains, since
`2¹ − 1 = 1`). Recorded with every run from here on.

**Read, part 2 — TREC-COVID's gap is query formulation, not the harness.**

TREC-COVID is the one BEIR dataset shipping several query formulations. `queries.jsonl`
carries the question form in `text`, and a keyword form and a narrative in `metadata`.
Indexing once and re-retrieving per field — retrieval is 0.1s, so this is nearly free:

| Query field | nDCG@10 | vs published 0.656 | recall@100 |
|---|---|---|---|
| `text` (BEIR canonical, what the loader uses) | 0.5644 | −0.0916 | 0.1088 |
| `metadata.query` (keyword form) | 0.5860 | −0.0700 | 0.1207 |
| `metadata.narrative` | 0.4554 | −0.2006 | 0.0870 |
| `query` + `text` | 0.6619 | +0.0059 | 0.1372 |
| `query` + `text` + `narrative` | 0.6970 | +0.0410 | 0.1377 |

Query formulation moves nDCG@10 on this dataset by **0.24** between its worst and best
form. That dwarfs tokenisation, `k1`/`b` and the gain function put together, and it is a
variable no entry in this log had been recording, because on every other BEIR dataset
there is only one query field and nothing to record.

**The loader keeps using `text`, and the reproduction is recorded as failed.**

`query + text` lands within 0.006 of the published figure and would turn this entry
green. That is precisely why it is not being adopted. Nothing independent establishes
that BEIR's Elasticsearch runs concatenated those fields; the only evidence for it is
that it matches the number being chased, which is the definition of fitting to the
target. `text` is what `queries.jsonl` designates as the query, it is what the loader
uses for every dataset, and it reproduces SciFact (+0.0152) and NFCorpus (−0.0026)
without special pleading. Changing the convention for the one dataset where the
convention is inconvenient would make all three numbers unfalsifiable.

So TREC-COVID stands as **not reproduced**, with the cause identified and quantified.
The harness is not the suspect: it reproduces two datasets either side of this one, and
the candidate explanation accounts for the entire gap on its own.

**Read, part 3 — where the headroom is.** Recall@100 is not comparable across these
datasets, because they differ enormously in how many relevant documents exist per query
(SciFact median 1, NFCorpus 16, TREC-COVID 478). Against the best recall@100 any system
could achieve given the judgements:

| Dataset | measured R@100 | oracle R@100 | share of ceiling |
|---|---|---|---|
| SciFact | 0.9220 | 1.0000 | **92.2%** |
| NFCorpus | 0.2461 | 0.9647 | **25.5%** |
| TREC-COVID | 0.1088 | 0.2674 | **40.7%** |

This reframes every remaining experiment. Dense retrieval, fusion and reranking all act
on the candidate set, and on SciFact BM25 has already found 92% of what is there — so a
win on SciFact is close to unavailable regardless of method quality, and any that
appears should be read as noise before it is read as progress. NFCorpus, at a quarter of
its ceiling, has room for a method to show something. TREC-COVID has room too, but with
50 queries it has very little statistical power, and that is worth saying before running
anything there rather than after.

**Next:** RM3 pseudo-relevance feedback — the first method in the project expected to
beat the baseline rather than define it. NFCorpus is the dataset to watch.

---

## Pending

Planned runs, in order. Each is a separate entry when it happens.

| # | Experiment | Settles |
|---|---|---|
| 1 | ~~BM25 on SciFact~~ | **Done** — nDCG@10 0.6802 vs published 0.665 |
| 2 | ~~Tokenisation ablation~~ | **Done** — stemming helps recall@100 (Holm p 0.018); stopwords do nothing |
| 3 | ~~`k1`/`b` sweep~~ | **Done** — tuned on train, no held-out gain (p 0.22); defaults kept |
| 4 | ~~BM25 on TREC-COVID and NFCorpus~~ | **Done** — NFCorpus reproduces; TREC-COVID does not (query formulation, −0.092) |
| 5 | Dense retrieval, same harness | The first real comparison |
| 6 | Hybrid via reciprocal rank fusion | Whether fusion beats both parents |
| 7 | Cross-encoder reranking over hybrid | Cost/benefit at depth 100 |
| 8 | Breakdown by query type | The actual question the project asks |

## Notes to self

- Graded qrels (TREC-COVID, NFCorpus) make the exponential-vs-linear gain choice
  matter. Record which was used in every entry from experiment 4 onward.
- Before claiming a hybrid win, check it survives a paired significance test. On 300
  queries, differences under about a point are noise.
- SciFact is a weak showcase for hybrid retrieval because BM25 is already strong on
  it. Expect thin gains and do not over-read them; TREC-COVID has the headroom.
