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

## Pending

Planned runs, in order. Each is a separate entry when it happens.

| # | Experiment | Settles |
|---|---|---|
| 1 | ~~BM25 on SciFact~~ | **Done** — nDCG@10 0.6802 vs published 0.665 |
| 2 | ~~Tokenisation ablation~~ | **Done** — stemming helps recall@100 (Holm p 0.018); stopwords do nothing |
| 3 | `k1`/`b` sweep | Whether BEIR's 0.9/0.4 is right for this corpus |
| 4 | BM25 on TREC-COVID and NFCorpus | Does the harness hold on graded qrels |
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
