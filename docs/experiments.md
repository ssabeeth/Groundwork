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

## Pending

Planned runs, in order. Each is a separate entry when it happens.

| # | Experiment | Settles |
|---|---|---|
| 1 | ~~BM25 on SciFact~~ | **Done** — nDCG@10 0.6802 vs published 0.665 |
| 2 | Tokenisation ablation: stem/no-stem, stopwords/none | How much of BM25's score is tokenisation |
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
