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

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

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

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

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

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

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

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

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

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

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

## 2026-09-21 — RM3 pseudo-relevance feedback

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

**Question:** Can a method beat the BM25 baseline, and does the headroom measured in
experiment 4 predict where?

**Setup:** RM3 over BM25 (`retrieval/rm3.py`). Retrieve, take the top `fb_docs`
documents, estimate a relevance model over their terms weighted by normalised BM25
score, keep the top `fb_terms`, interpolate with the original query at weight `alpha`,
and re-retrieve. No relevance judgements are used — the assumption is that the top
documents are relevant, which is exactly the assumption that can fail.

Three parameters, so experiment 3's discipline applies: a 96-cell grid
(`fb_docs` × `fb_terms` × `alpha`) swept on **train**, the winner scored once on
**test**. SciFact train is 809 queries, NFCorpus train 2,590. BM25 stays at
`k1=0.9, b=0.4` with Porter stemming and Lucene stopwords. Paired randomisation test,
100,000 resamples, seed 0, Holm across the two datasets per metric.

**Result:**

| Dataset | tuned setting | BM25 | RM3 | delta | Holm p |
|---|---|---|---|---|---|
| NFCorpus | `fb_docs=5, fb_terms=50, alpha=0.8` | 0.3224 | **0.3433** | **+0.0208** | **0.0005** |
| SciFact | `fb_docs=20, fb_terms=20, alpha=0.2` | 0.6802 | 0.6848 | +0.0046 | 0.290 |

nDCG@10 above; recall@100 below:

| Dataset | BM25 | RM3 | delta | Holm p |
|---|---|---|---|---|
| NFCorpus | 0.2461 | **0.3105** | **+0.0645** | **<0.0001** |
| SciFact | 0.9220 | 0.9253 | +0.0033 | 1.000 |

**Read:** **The first method in this project to beat its baseline, and it beats it on
exactly the dataset experiment 4 predicted.**

That prediction is the part worth keeping. Experiment 4 measured how much of the
achievable recall@100 BM25 was already capturing: 92.2% on SciFact, 25.5% on NFCorpus.
The inference drawn there — that a better method has almost nothing to win on SciFact
however good it is, and room to win on NFCorpus — was made before RM3 existed. It held.
NFCorpus gains 0.0208 nDCG@10 at Holm p 0.0005 and 0.0645 recall@100 at the floor of
what 100,000 resamples can report; SciFact gains 0.0046 at p 0.29, which is nothing.
RM3 moves NFCorpus from 25.5% of its recall ceiling to 32.2%.

The tuned parameters say the same thing from the other side, and neither was chosen by
hand. SciFact's sweep settled on `alpha=0.2` — keep 80% of the original query, barely
expand. NFCorpus settled on `alpha=0.8` with fifty expansion terms — largely replace the
query with terms harvested from the feedback documents. Two corpora, the same grid, and
opposite answers about how much to trust the original query.

The mechanism is not mysterious. NFCorpus queries are short consumer-health phrases
against medical writing, and the vocabulary mismatch between the two is exactly what
feedback terms repair. SciFact queries are already scientific claims written in the
register of the documents they are matched against, so there is little mismatch to
repair and expansion mostly adds noise. This is the first concrete evidence for the
project's central claim — that retrieval method is query-dependent — though it is
evidence at the level of *corpora* rather than individual queries, which is the weaker
version. Experiment 8 is where the per-query version gets tested.

**What it costs.** Retrieval time on SciFact goes from 0.1s to 9.7s for 300 queries —
two passes plus re-tokenising twenty feedback documents per query. Still fast in
absolute terms, but roughly a hundredfold, and worth stating next to a two-point gain.
Feedback documents are re-tokenised on demand and cached; the cache is bounded by
`fb_docs × queries`, not corpus size, which is what makes a 96-cell sweep finish in
181 seconds instead of re-tokenising the same documents ten thousand times.

**A caution on the SciFact result.** +0.0046 at p 0.29 is not a small win, it is no
measurable win. It would be easy to report it as "RM3 helps slightly on SciFact too".
264 of 300 queries are unchanged, and of those that move, the direction is close to even.
The honest reading is that SciFact had nothing left to give, which is what its recall
ceiling already said.

**Next:** Dense retrieval on the same harness. NFCorpus is the dataset to watch for the
same reason it was here. TREC-COVID stays out of method comparisons for now: 50 queries
gives too little power to distinguish anything, and its reproduction is already known
to be off.

---

## 2026-09-21 — Dense retrieval

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

**Question:** Does a bi-encoder beat BM25 on this harness, and what does it cost?

**Setup:** `sentence-transformers/all-MiniLM-L6-v2`, mean pooling, L2-normalised so the
inner product is cosine similarity, maximum sequence length 256 word pieces. Embeddings
cached to gitignored `data/embeddings/`, keyed by model, length and a digest of the
corpus. SciFact and NFCorpus test splits, depth 100. Dense is an optional extra: core
install stays numpy and tqdm.

**Result:**

| Dataset | metric | BM25 | dense | delta | p |
|---|---|---|---|---|---|
| NFCorpus | nDCG@10 | 0.3224 | 0.3173 | −0.0051 | 0.655 |
| NFCorpus | recall@100 | 0.2461 | **0.3115** | **+0.0654** | **<0.0001** |
| SciFact | nDCG@10 | 0.6802 | 0.6451 | −0.0351 | 0.064 |
| SciFact | recall@100 | 0.9220 | 0.9250 | +0.0030 | 0.885 |

Encoding: 18.8s for NFCorpus, 25.0s for SciFact, on MPS.

**Read:** **Dense retrieval never beats BM25 at ranking here, and on NFCorpus it finds
substantially more.** That combination is the interesting part. On NFCorpus the two are
indistinguishable at nDCG@10 (p 0.655) while dense recovers 6.5 points more of the
relevant set by depth 100 — it is locating documents BM25 misses entirely and then
failing to rank them above BM25's own hits. That is the profile of a system that should
fuse well, which is what the next experiment tests.

**The truncation number belongs next to every one of these figures. 78.8% of NFCorpus
documents and 71.0% of SciFact documents exceed the model's 256-token limit** and are
silently cut off. So this is not a measurement of "dense retrieval on scientific
abstracts"; it is a measurement of dense retrieval on the first 256 word pieces of them.
Whether the remaining text would help is untested and would need either a
longer-context encoder or chunking. Reporting the nDCG without this would make the
comparison look like a property of dense retrieval rather than a property of this
configuration, which is why `describe()` records it on every run.

The model is also a variable, not a constant. all-MiniLM-L6-v2 is a small general-purpose
encoder, not a scientific-domain one, and it was chosen for being cheap rather than for
being the strongest available. A stronger or domain-matched model would move these
numbers, possibly a lot. What is claimed here is only what was measured.

**Next:** Fuse it with BM25.

---

## 2026-09-21 — Hybrid retrieval by reciprocal rank fusion

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

**Question:** Does fusing BM25 and dense beat both parents, and does it beat the much
cheaper RM3?

**Setup:** Reciprocal rank fusion, `RRF(d) = Σ 1/(k + rank(d))`, which discards scores
and keeps only positions — so no score normalisation has to be invented between BM25's
unbounded sums and cosine similarities. `k` is swept on **train** and the chosen value
scored once on **test**, because tuning a fusion parameter on the evaluation split is
the most common way a hybrid result is overstated. Paired randomisation tests with Holm
across the four comparisons per dataset per metric.

**Result:**

Swept on train, the best `k` was **10 on NFCorpus and 1 on SciFact** — both far below
the conventional default of 60, which would have cost 0.005 and 0.013 nDCG@10
respectively. The default is a default, not a constant.

nDCG@10 on test:

| Dataset | BM25 | dense | RM3 | **RRF** |
|---|---|---|---|---|
| NFCorpus | 0.3224 | 0.3173 | 0.3433 | **0.3559** |
| SciFact | 0.6802 | 0.6451 | 0.6848 | **0.7146** |

Fusion against each parent, Holm-adjusted:

| Dataset | comparison | delta nDCG@10 | Holm p |
|---|---|---|---|
| NFCorpus | RRF vs BM25 | +0.0334 | **<0.0001** |
| NFCorpus | RRF vs dense | +0.0386 | **<0.0001** |
| NFCorpus | RRF vs RM3 | +0.0126 | 0.290 |
| SciFact | RRF vs BM25 | +0.0344 | **0.0021** |
| SciFact | RRF vs dense | +0.0695 | **<0.0001** |
| SciFact | RRF vs RM3 | +0.0298 | **0.0056** |

Recall@100 rises with it: NFCorpus 0.2461 → 0.3217, SciFact 0.9220 → 0.9550.

**Read:** **Fusion beats both parents on both datasets, and this is the project's
strongest result.** It also beats each parent by more than the parents differ from each
other, which is the signature of the two systems making uncorrelated errors rather than
one simply being better.

**But the comparison that matters commercially is RRF against RM3, and on NFCorpus it is
not significant** (+0.0126, Holm p 0.290). RM3 needs numpy, runs in seconds, and has no
model to download, no GPU, and no 2GB dependency tree. Fusion needs all of that and, on
that dataset, cannot be shown to do better. On SciFact fusion does win over RM3 (+0.0298,
Holm p 0.0056). So the honest summary is that fusion is the better method where it has
been measured, and that on one of two datasets a far cheaper method was statistically
indistinguishable from it — which is worth knowing before anyone deploys a GPU to serve
it.

**A correction to experiment 4.** That entry inferred from the recall-ceiling table that
SciFact, at 92.2% of achievable recall@100, had "almost nothing to win... whatever the
method". RM3 confirmed it. Fusion refutes it: +0.0344 nDCG@10 at Holm p 0.0021, and
recall@100 up to 0.9550. The inference was too broad. A recall ceiling bounds what
*recall-limited* methods can gain; it says nothing about improvements to the *ordering*
of documents already retrieved, and SciFact's nDCG@10 of 0.68 left plenty of room there.
The prediction was right about RM3 for the right reason and wrong about fusion for a
reason the original argument did not consider. Recorded rather than quietly amended.

**Next:** The per-query question the project was built to ask.

---

## 2026-09-21 — Is retrieval method query-dependent?

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

**Question:** The README's central claim. Does lexical retrieval win on lexically
specific queries and lose on the rest, or is one method simply better?

**Setup and pre-specification.** The hypothesis was fixed before any correlation was
computed, and stated in `scripts/analyse_queries.py` before it was run:

> per-query (nDCG@10 of BM25 − nDCG@10 of dense) correlates **positively** with the
> rarity of the query's rarest term, measured as `max_idf` from the BM25 index.

The mechanism it encodes: BM25 rewards exact matches weighted by rarity, so a query
containing a genuinely rare token hands it something close to a unique key, while a
bi-encoder maps that token into a neighbourhood of things keeping similar company.

One continuous predictor rather than query-type buckets, deliberately. Labelling queries
by type and comparing group means invites choosing the labelling that shows an effect,
and with enough candidate groupings one always will. `max_idf` is computable from the
query and the index alone, before any retrieval is run, which rules out circularity.
`mean_idf`, out-of-vocabulary rate and query length are reported as **secondary** and
labelled as such, Holm-adjusted among themselves, because they were not pre-specified.

Significance is a permutation test on the pairing, 20,000 permutations, seed 0.

**Result:**

| Dataset | queries | mean advantage | rho (`max_idf`) | p | Holm across datasets |
|---|---|---|---|---|---|
| NFCorpus | 323 | +0.0051 | **+0.1578** | 0.0046 | **0.0092** |
| SciFact | 300 | +0.0351 | **+0.1191** | 0.0362 | **0.0362** |

Mean BM25-minus-dense advantage by `max_idf` tercile:

| Tercile | NFCorpus | SciFact |
|---|---|---|
| lowest `max_idf` | **−0.0169** | **−0.0240** |
| middle | −0.0002 | +0.0560 |
| highest `max_idf` | **+0.0327** | **+0.0733** |

**Read:** **The claim holds, on both datasets, in the predicted direction, and it is
invisible in the aggregate.**

NFCorpus is the clean demonstration. Compare the two systems the ordinary way and the
answer is "no difference": BM25 0.3224 against dense 0.3173, p 0.655, a mean per-query
advantage of +0.005. Split the same 323 queries by how rare their rarest term is and the
answer becomes "it depends, and systematically so": dense is ahead by 0.017 on the least
lexically specific third and behind by 0.033 on the most specific third. The aggregate
comparison was not wrong, it was answering a question whose true answer is an average of
two opposite effects.

The correlations are modest — rho around 0.12 to 0.16, so lexical specificity explains a
small share of the variance in which system wins. That is worth stating plainly rather
than rounding up. Many other things determine per-query outcomes. But the effect is in
the predicted direction on two independent datasets, survives Holm adjustment across
them, and the terciles are monotone on both, which is more than a marginal correlation
alone would establish.

One secondary result is worth flagging *as* secondary, because it was not predicted:
query length correlates negatively with BM25's advantage on NFCorpus (rho −0.2219,
Holm p 0.0004) and more strongly than the primary predictor does positively. Longer
queries favour the encoder. That is plausible after the fact — more context to embed,
more terms to dilute an IDF-weighted sum — but it was not the hypothesis, it was found
by looking, and it should be treated as a lead for a pre-registered test on a third
dataset rather than as an established finding. Recording the distinction is the only
thing that keeps the primary result meaningful.

**What this does not show.** Correlation across queries, not causation, and a single
encoder at a 256-token limit that truncates 71-79% of documents. Whether the effect
survives a longer-context or domain-matched model is untested and is the obvious next
thing to break.

**Next:** Cross-encoder reranking over the fused candidate set, bounded by the
recall@100 now measured at 0.3217 on NFCorpus and 0.9550 on SciFact.

---

## 2026-09-21 — Cross-encoder reranking over the fused candidates

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

**Question:** Does rescoring the fused shortlist with a cross-encoder earn its cost?

**Setup:** `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 100 fused candidates,
applied to the tuned RRF runs from experiment 6. A cross-encoder concatenates query and
document and runs the model over both together, so nothing can be precomputed and the
cost scales with candidates times queries. Paired randomisation test against the
un-reranked fused run, Holm across the two datasets.

**Result:**

| Dataset | fused | reranked | delta | p | Holm |
|---|---|---|---|---|---|
| SciFact | 0.7146 | 0.6865 | **−0.0281** | 0.047 | 0.094 |
| NFCorpus | 0.3559 | 0.3554 | −0.0005 | 0.936 | 0.936 |

Cost: 288s for 30,000 pairs on SciFact, 344s for 32,300 on NFCorpus, on MPS. Fusion
itself takes seconds.

recall@100 is unchanged on both, necessarily: reranking reorders a candidate set and
cannot add to it. The ceilings measured in experiment 6 — 0.9550 and 0.3217 — are
untouched and untouchable by this stage.

**Read:** **The most expensive method in the project buys nothing, and on SciFact the
point estimate is negative.** The SciFact drop has a raw p of 0.047 and does not survive
Holm across the two datasets, so the honest statement is "no evidence of benefit, and a
suggestion of harm worth checking on a third dataset" rather than "reranking hurts".
What can be said without hedging is that six minutes of GPU time per dataset produced no
measurable improvement over a fusion that took seconds.

A result I nearly reported and should not have: nDCG@1 on NFCorpus rises from 0.4613 to
0.4871 after reranking, which looks like the expected "helps the very top of the
ranking" story. It does not survive a paired test (+0.0258, p 0.2077, Holm 0.4155). The
aggregate moved because a handful of queries did, and 276 of 323 are unchanged. Testing
it was the difference between a finding and an anecdote.

**Why this is the expected direction on reflection.** `ms-marco-MiniLM-L-6-v2` is trained
on MS MARCO — real web search queries against web passages. SciFact queries are
scientific claims and NFCorpus queries are consumer-health phrases against medical
writing. Neither resembles the training distribution, and a cross-encoder's whole
advantage is that it reads the pair jointly, which is exactly the part that transfers
worst when the pair looks unfamiliar. The bi-encoder in experiment 5 had the same
problem and it showed up the same way.

**What this does not establish.** That reranking is useless — only that *this*
reranker, at depth 100, on these two datasets, is. A cross-encoder trained on scientific
or biomedical text is the obvious next thing to try, and it is a cheap experiment now
that the harness and the ceilings exist. The claim here is bounded by what was measured,
as with the bi-encoder.

**Next:** The remaining pending entries are a domain-matched encoder for experiments 5
and 7, and TREC-COVID's query-formulation question from experiment 4.

---

## Pending

Planned runs, in order. Each is a separate entry when it happens.

| # | Experiment | Settles |
|---|---|---|
| 1 | ~~BM25 on SciFact~~ | **Done** — nDCG@10 0.6802 vs published 0.665 |
| 2 | ~~Tokenisation ablation~~ | **Done** — stemming helps recall@100 (Holm p 0.018); stopwords do nothing |
| 3 | ~~`k1`/`b` sweep~~ | **Done** — tuned on train, no held-out gain (p 0.22); defaults kept |
| 4 | ~~BM25 on TREC-COVID and NFCorpus~~ | **Done** — NFCorpus reproduces; TREC-COVID does not (query formulation, −0.092) |
| 4b | ~~RM3 pseudo-relevance feedback~~ | **Done** — +0.0208 nDCG@10 on NFCorpus (Holm p 0.0005); nothing on SciFact |
| 5 | ~~Dense retrieval~~ | **Done** — never beats BM25 at ranking; +0.065 recall@100 on NFCorpus; 71-79% truncated |
| 6 | ~~Hybrid via reciprocal rank fusion~~ | **Done** — beats both parents on both datasets; ties RM3 on NFCorpus |
| 7 | ~~Cross-encoder reranking over hybrid~~ | **Done** — no measurable gain; 6 min GPU for nothing |
| 8 | ~~Breakdown by query type~~ | **Done** — lexical advantage rises with query term rarity, both datasets |

## 2026-09-21 — Encoder choice, chunking, and a falsification test of experiment 8

> **Pre-migration entry.** The BM25 figures below index title and body concatenated as
> one field. Experiment 10 established that BEIR indexes them separately, and every
> figure here has since been re-run on that basis — see *The migration re-run* for the
> new numbers and for which conclusions moved. This entry is left as it was written: the
> runs it describes are real and are preserved in `results/` under `-singlefield`.

**Question:** Experiments 5 and 7 concluded that dense retrieval and reranking do not
beat BM25 here. Both used models trained on web search text, reading documents truncated
at 256 word pieces. Was that a finding about dense retrieval, or about one weak
configuration? And does experiment 8's query-dependence result survive a competent
encoder?

The second question is the important one. Experiment 8's correlation compares BM25
against dense per query. If the encoder is weak, "BM25 wins where the query has rare
terms" and "BM25 wins where the weak model fails" are indistinguishable, and the result
could be an artifact of model quality rather than a fact about lexical matching. This
experiment was designed to be able to overturn it.

**Setup:** Four encoders on NFCorpus, all with the same harness, depth and tokenisation
as before: all-MiniLM-L6-v2 (the original, 256 tokens), all-mpnet-base-v2 (384),
S-PubMedBert-MS-MARCO (350, biomedical), bge-small-en-v1.5 (512). Then chunking — 200-word
windows with 50 overlap, scoring a document by its best-matching window — on the two
strongest. The winner was then run on SciFact, fused with BM25 (`k` swept on train as
always), and put through experiment 8's correlation unchanged.

**Result — encoders, NFCorpus nDCG@10:**

| Encoder | context | truncated | nDCG@10 | recall@100 |
|---|---|---|---|---|
| BM25 (reference) | — | — | 0.3224 | 0.2461 |
| all-MiniLM-L6-v2 (experiment 5) | 256 | 78.8% | 0.3173 | 0.3115 |
| S-PubMedBert-MS-MARCO | 350 | 37.8% | 0.3142 | 0.2921 |
| all-mpnet-base-v2 | 384 | 40.0% | 0.3346 | 0.3385 |
| **bge-small-en-v1.5** | 512 | 9.1% | **0.3391** | 0.3059 |

Chunking, which removes truncation almost entirely:

| Configuration | truncated | nDCG@10 | paired vs unchunked |
|---|---|---|---|
| bge-small | 9.1% | **0.3391** | — |
| bge-small, chunked | 0.0% | 0.3348 | −0.0043, p 0.256 |
| mpnet | 40.0% | **0.3346** | — |
| mpnet, chunked | 2.1% | 0.3225 | −0.0120, p 0.026 (Holm 0.052) |

**Result — the falsification test.** Experiment 8's correlation, re-run against
bge-small instead of MiniLM, everything else identical:

| Dataset | encoder | mean advantage | rho (`max_idf`) | p | Holm |
|---|---|---|---|---|---|
| NFCorpus | MiniLM (exp. 8) | +0.0051 | +0.1578 | 0.0046 | 0.0092 |
| NFCorpus | **bge-small** | −0.0167 | **+0.0916** | **0.0990** | 0.0990 |
| SciFact | MiniLM (exp. 8) | +0.0351 | +0.1191 | 0.0362 | 0.0362 |
| SciFact | **bge-small** | −0.0398 | **+0.1550** | **0.0077** | 0.0154 |

Terciles under bge-small remain monotone on both: NFCorpus −0.0315 / −0.0255 / +0.0071,
SciFact −0.1023 / −0.0162 / −0.0010.

**Read — three corrections, in descending order of how wrong I was.**

*Experiment 5's headline conclusion does not survive.* It said dense retrieval "never
beats BM25 at ranking". With bge-small it does, on both datasets: SciFact 0.7200 against
0.6802 (+0.0398, raw p 0.023), NFCorpus 0.3391 against 0.3224 (+0.0167, p 0.120). Neither
clears Holm across the family of four comparisons, so the honest claim is not "dense
beats BM25" — it is that **the previous claim is no longer supportable**, because the
point estimates changed sign and the margin on SciFact is the largest single-system gap
measured in this project. On SciFact bge-small alone (0.7200) also matches the entire
BM25 + MiniLM fusion from experiment 6 (0.7146, p 0.70). One better encoder was worth
more than fusing two worse systems.

*Domain matching did not help; general model quality did.* The biomedical encoder was
the **worst** of the three strong candidates on a nutrition-and-medicine corpus — 0.3142,
below BM25 and below both general models. This was the opposite of what I predicted when
proposing the experiment. Whatever S-PubMedBert gains from biomedical pretraining, it
loses to bge-small's retrieval training, and "use a domain model for a domain corpus" is
not supported by anything measured here.

*Truncation was the wrong thing to worry about.* Experiment 5 flagged 78.8% truncation
as a caveat attached to every dense number, and it sounded damning. But eliminating it by
chunking does not help. Paired, unchunked is ahead by 0.0043 on bge-small (p 0.256, no
measurable difference) and by 0.0120 on mpnet (p 0.026 raw, Holm 0.052 — borderline, and
in the direction of chunking being *worse*). So the discarded tails were not carrying
signal, and max-pooling over windows gives one off-topic passage a chance to match
spuriously. The caveat was reasonable to state and wrong about what was limiting
performance — which is why it was worth testing rather than assuming in either direction. Context length still correlates with
quality across encoders, but that is confounded with everything else those models differ
in, and this experiment cannot separate the two.

**Read — what happened to experiment 8.** The claim is weakened but not overturned, and
the distinction matters.

What replicates: the direction is positive under both encoders on both datasets, and the
terciles are monotone in all four combinations. BM25's relative advantage does rise with
query term rarity.

What does not: significance is encoder-dependent. With MiniLM, NFCorpus was the stronger
result (p 0.0046) and SciFact the weaker (p 0.0362). With bge-small they swap — SciFact
holds at p 0.0077, NFCorpus falls to p 0.0990 and no longer clears 0.05. Since the
underlying quantity should not depend on which encoder is used to estimate it, **the
effect size is not robustly estimated, and experiment 8's numbers overstated it.** The
README's confident framing of that result has been rewritten accordingly.

The mean advantage flipping sign on both datasets — BM25 was ahead on average under
MiniLM, behind under bge-small — is the clearest illustration of why. Experiment 8 was
measuring a mixture of "lexical specificity favours BM25" and "this encoder is weak", and
only the falsification test could separate them. It partially did: the effect is real in
direction and about half the size originally claimed.

**Result — fusion, rebuilt on the better encoder.** Best configurations measured:

| Dataset | best single | RRF(BM25 + bge) | vs best single | vs BM25 |
|---|---|---|---|---|
| SciFact | bge 0.7200 | **0.7399** | +0.0199, p 0.079 | +0.0597, Holm <0.0001 |
| NFCorpus | RM3 0.3433 | **0.3659** | +0.0268, Holm 0.0001 | +0.0435, Holm <0.0001 |

Fusion still produces the best number on both datasets, and still beats BM25 decisively.
But on SciFact it no longer beats its own strongest parent significantly (p 0.079), which
is a weaker claim than experiment 6 was able to make when both parents were mediocre.

**Next:** TREC-COVID's query formulation, then a third dataset — which now has a clearer
job than it did yesterday, since it is the only way to settle whether the query-dependence
effect is real at the size the SciFact runs suggest or the size the NFCorpus runs suggest.

---

## 2026-09-21 — Multi-field indexing, and why TREC-COVID would not reproduce

**Question:** Experiment 4 left TREC-COVID as a failed reproduction, 0.0916 below the
published figure, and attributed it to query formulation. Was that diagnosis right?

**Setup:** Read BEIR's paper rather than reasoning further from inside the repository.
Two sentences settle it, and both contradict this project's own documentation:

> "We use Anserini with the default Lucene parameters (k=0.9 and b=0.4)."

> "We index the title (if available) and passage as separate fields for documents."

The first corrects an attribution: `decisions.md` said the published baselines came from
Elasticsearch. They come from Anserini. The paper adds that they "also tested
Elasticsearch BM25 and Anserini + RM3 expansion, but found Anserini BM25 to perform the
best". Anserini is Lucene-based, so the argument for the Lucene IDF variant survives
intact — but the reason recorded for making the choice was factually wrong.

The second is the substantive one. This project indexes `title + " " + text` as a single
bag of words, and the `bm25.py` docstring asserted that this matched BEIR's convention.
It does not. Under Lucene, separate fields mean each field carries its own document
lengths, its own average length and its own document frequencies; a query scores against
each independently and the scores are summed. Concatenation collapses all of that into
one distribution.

`MultiFieldBM25Retriever` implements the BEIR arrangement. Same tokenisation, same
`k1=0.9, b=0.4`, same everything else.

**Result:**

| Dataset | concatenated | delta | multi-field | delta | published |
|---|---|---|---|---|---|
| SciFact | 0.6802 | +0.0152 | **0.6636** | **−0.0014** | 0.665 |
| NFCorpus | 0.3224 | −0.0026 | **0.3253** | **+0.0003** | 0.325 |
| TREC-COVID | 0.5644 | −0.0916 ✗ | **0.6362** | **−0.0198** ✓ | 0.656 |

**Read:** **All three datasets now reproduce, and the failure was never about queries.**

The magnitudes are hard to argue with. On SciFact the gap to the published figure goes
from +0.0152 to −0.0014, and on NFCorpus from −0.0026 to +0.0003 — both within a
thousandth of numbers produced by different software on different hardware five years
earlier. That is not the accuracy a reimplementation gets by coincidence. On TREC-COVID
the gap closes from −0.0916, three times the tolerance, to −0.0198, inside it.

The mechanism is the one the field structure predicts, and TREC-COVID is where it bites
hardest because 24.6% of its documents have empty abstracts. Under concatenation a
title-only document is a very short *document*, and BM25's length normalisation inflates
whatever it matches — 42,140 documents getting an unearned boost. Split into fields, the
same document is an ordinary-length title plus an empty body contributing nothing.
SciFact and NFCorpus have almost no title-only documents, which is exactly why they
reproduced tolerably under concatenation and TREC-COVID did not.

**Experiment 4's diagnosis was wrong, and the discipline it applied was right.** That
entry identified query formulation as the cause, having measured a 0.24 spread across
formulations, and noted that `query + text` would land within 0.006 of the published
figure. It then refused to adopt it, on the grounds that nothing independent established
BEIR had done that and the only argument for it was that it matched the target.

That refusal is now vindicated in the strongest available way. Adopting `query + text`
would have produced a number within 0.006 of the published one **for entirely the wrong
reason**, closed the investigation, and left the real defect — a baseline that did not
match the method it claimed to match — in place across every dataset and every
experiment built on top of it. The right number by the wrong route would have been worse
than the honest failure, because the honest failure kept the question open until the
actual cause turned up.

The query-formulation measurement itself stands: the spread across formulations really
is 0.2416 nDCG@10 and is worth knowing. It was simply not the explanation for this gap.

**What is not being changed yet.** Concatenation remains the default. Every result in
this repository — the tokenisation ablation, the `k1`/`b` sweep, RM3, dense, fusion,
reranking, the query-dependence analysis — was produced with it, and switching the
default silently would invalidate all of them at once while leaving the prose describing
them intact. The comparisons between methods are internally consistent under
concatenation, and there is no reason to think the qualitative findings depend on it,
but "no reason to think" is not a measurement.

So this is recorded as a decision to take deliberately rather than a fix to apply
quietly. The options, with the trade-off stated:

- **Switch the default and re-run everything.** Correct, and makes every published number
  comparable to BEIR. Costs a full re-run of nine experiments, including the two that
  take six minutes of GPU each and the three that index 171,332 documents.
- **Keep concatenation for method comparisons, use multi-field for reproduction claims.**
  Cheaper and defensible — the baseline-versus-published question and the
  method-versus-method question are different questions — but it means the repository
  contains two BM25 baselines, which is exactly the kind of ambiguity this project exists
  to avoid.

**Next:** That decision, then a third dataset.

---

## 2026-09-21 — The migration re-run: every BM25 number, again

Experiment 10 established that BEIR indexes title and passage as separate fields and
that doing the same reproduces the published baselines. This entry is the consequence:
`MultiFieldBM25Retriever` is now the default in `run_baseline.py`, `run_sweep.py`,
`run_rm3.py`, `run_hybrid.py` and `diagnose_query_fields.py`, `--single-field` restores
concatenation, and **every BM25-derived result in this repository was re-run**.

Dense-only results are untouched, because they never see a BM25 index. Everything else
moved, including two conclusions.

**Why the earlier entries were not rewritten.** Each pre-migration entry now carries a
banner saying what it is and where the new numbers are, and is otherwise left exactly as
written. Editing nine entries to show numbers they were not based on would produce a log
in which this project had never been wrong, which is the opposite of the point. The runs
those entries describe are preserved in `results/` under a `-singlefield` suffix.

### Baselines against the published figures

| Dataset | Concatenated | Multi-field | Published | Multi-field delta |
|---|---|---|---|---|
| SciFact | 0.6802 | 0.6636 | 0.665 | −0.0014 |
| NFCorpus | 0.3224 | 0.3253 | 0.325 | +0.0003 |
| TREC-COVID | 0.5644 | 0.6362 | 0.656 | −0.0198 |
| SciDocs | — | 0.1577 | 0.158 | −0.0003 |

Three of the four land within 0.0014 of figures produced by different software years
earlier. SciFact's baseline went *down* and became more correct: the concatenated run sat
at +0.0152 from the published figure and the multi-field run sits at −0.0014. That +0.0152
had been read as a tokenisation difference pointing the expected way — Porter stemming and
a stopword list normalise more aggressively than the reference analyzer. Part of it was an
indexing error pointing the flattering way. A reimplementation scoring *above* the thing it
reimplements deserved more suspicion than it got, and the plausible explanation available
for it is exactly why it did not get any.

### Two conclusions reversed

**Experiment 2's finding swapped metrics entirely.** It reported that stemming was a
significant win on recall@100 and undetermined on nDCG@10. Under multi-field indexing it
is the exact opposite:

| Stemming, SciFact | Concatenated | Multi-field |
|---|---|---|
| nDCG@10 | +0.0175, Holm 0.1661 — undetermined | **+0.0309, Holm 0.0156 — significant** |
| recall@100 | +0.0361, Holm 0.0183 — significant | **+0.0083, Holm 1.0000 — undetermined** |

The mechanism is visible in the numbers. Concatenation gives every document one length,
so a stemmed match on a title term and a stemmed match deep in a body are normalised
identically; stemming's benefit showed up as *finding* documents it otherwise missed,
which is recall. Scoring the title as its own short field makes a stemmed title match
worth much more at the top of the ranking, which is nDCG. Same tokenisation, same corpus,
opposite conclusion, entirely because of how the index was built.

Stopword removal remains worth nothing on either metric (Holm 0.4608 on nDCG@10, 1.0000
on recall@100), which is the one part of experiment 2 that survived untouched.

**Fusion stopped being worth anything on SciFact.** Against a strong encoder, RRF now
gains +0.0007 nDCG@10 over bge-small alone (p 0.9413). Before the migration it gained
+0.0199 (p 0.0793) — never significant, but enough to describe as a hybrid win. It is now
unambiguously nothing. On NFCorpus fusion still earns its place: +0.0219 over bge-small
(Holm p 0.0010).

The reason is worth stating because it is counter-intuitive. A *better* BM25 made fusion
*less* useful. Fusion pays off in proportion to how differently its inputs fail, and the
concatenated index failed in ways an encoder does not — inflating short title-only
documents — which made it accidentally complementary. Correcting it made BM25 better and
more similar to the encoder at the same time, and the second effect was larger. Component
quality and component diversity are different things, and only one of them was being
measured.

### One conclusion strengthened

Experiment 9 overturned experiment 5 but could only say "the earlier claim is not
supportable", because bge-small beating BM25 on SciFact did not clear Holm correction
(+0.0398, Holm 0.0929). It now does: **+0.0564, Holm p 0.0036**. Dense retrieval beating
BM25 on SciFact is a positive finding rather than the absence of a negative one.

### What did not move

- **RM3's split.** NFCorpus +0.0188 (Holm 0.0006), SciFact −0.0086 (p 0.1168). The point
  estimate on SciFact crossed zero — it was +0.0046 — but both readings are null and the
  conclusion is the one experiment 4b's recall-ceiling argument predicted.
- **`k1`/`b` tuning.** +0.0028 on held-out test at p 0.4149, against +0.0063 at p 0.217
  before. Still nothing, and the defaults still stay.
- **Query-dependence.** All four dataset-and-encoder cells moved by less than 0.02 in rho.
  The effect is robust to the indexing change, which is worth knowing given how much else
  was not.


### A bug the migration itself introduced, found four hours later

`run_sweep.py` was not migrated with the others, and the checklist in this log recorded
that it had been. It could not be: sweeping without re-indexing needs
`with_parameters`, and `MultiFieldBM25Retriever` did not have that method, so the script
still constructed a `BM25Retriever`. The re-run therefore produced a `k1`/`b` sweep that
was bit-identical to the pre-migration one — and the parameters it chose were then
applied to a *multi-field* test run, so the tuned figure was tuned on one index and scored
on another.

Two tests should have caught it and neither could, for the same reason: **sweep records
did not describe their retriever.** The filename check reads the retriever's `describe()`
and found nothing to read; the settings check skipped sweep files entirely because they
have no top-level `metrics` block. A record that says nothing cannot contradict itself.

Fixed by adding `MultiFieldBM25Retriever.with_parameters` with tests against a
freshly-indexed retriever, recording `retriever.describe()` in both the `k1`/`b` and RM3
sweep records, and extending the settings check to any file that scores anything rather
than only files shaped like a baseline run. Preserved pre-migration archives are held to a
different requirement instead — they must carry a note saying what they are — because
back-filling a description onto them would mean writing down settings nothing observed.

The generalisable lesson is not about sweeps. It is that a check which reads a field is
worthless against a file that omits the field, so "records its settings" and "its settings
are consistent" have to be enforced together or neither holds.

---

## Experiment 11 (pre-registration): SciDocs as a third dataset

**This entry was written and committed before the dataset was retrieved even once.**
Everything below the line is a prediction. The git history is the evidence: if the
numbers disagree with what is written here, the prediction was wrong and stays on the
page. That is the whole point of writing it first.

Experiment 8 found that BM25's per-query advantage over a dense encoder rises with the
rarity of the query's rarest term. Experiment 9 attacked it with a better encoder and it
half-survived — consistent in direction across two encoders and two corpora, but which
dataset clears p < 0.05 depends on which encoder measures it. Two datasets of ~300
queries cannot separate "small real effect" from "noise with a consistent sign".

**Why SciDocs.** 25,657 documents, 1,000 test queries, binary qrels, and a published
BM25 baseline of 0.158 (Thakur et al. 2021, Table 2). Three properties earn it the slot:

- **Power.** 1,000 queries is more than SciFact and NFCorpus combined. If the effect is
  the size those two suggest, this is enough to detect it; if nothing shows up here,
  that is informative rather than inconclusive.
- **Binary qrels**, like SciFact, so exponential and linear gain coincide and the gain
  choice cannot confound anything.
- **It is hostile to the hypothesis.** SciDocs queries are paper titles and relevance is
  citation-based — "papers this paper cites" is a semantic relation, not a lexical one,
  which is why BM25 scores 0.158 there against 0.665 on SciFact. A predictor of *BM25's
  relative advantage* is being tested in the regime where BM25 is weakest overall. That
  is the condition under which it is most likely to fail, which is why it is worth
  running.

**No train split.** BEIR ships qrels for SciDocs test only. Nothing here needs tuning:
the baseline uses BEIR's `k1=0.9, b=0.4` and the query analysis has no free parameters.
Fusion is run at `k=60`, the conventional default, fixed here in advance and explicitly
untuned — not swept, and not to be swept on test later.

---

**H1 (primary, carried over unchanged from experiment 8).** Per-query
`nDCG@10(BM25) − nDCG@10(dense)` correlates **positively** with `max_idf`.
Prior point estimates: +0.158 and +0.119 (MiniLM), +0.092 and +0.155 (bge-small).
Predicted here: positive, and significant at p < 0.05 given 1,000 queries.

**H2 (promoted from secondary, and this is its first honest test).** Per-query advantage
correlates **negatively** with query length (`num_terms`). This was *found by looking* on
NFCorpus (rho −0.2219, Holm p 0.0004) and has never been tested on data that did not
generate it. Predicted: negative. A result here counts; the NFCorpus one never did.

**H3 (harness check).** Multi-field BM25 lands within 0.03 of the published 0.158.

One caveat on H3 that the tolerance rule does not capture: `REFERENCE_TOLERANCE` is
absolute, so ±0.03 is ±4.5% of SciFact's 0.665 but ±19% of SciDocs' 0.158. Passing on
SciDocs is therefore much weaker evidence than passing on SciFact, and should not be
read as an equally strong reproduction. The rule is left alone rather than tuned to
taste after seeing which way the number went.

**Two further expectations, recorded so they can be wrong.** Dense retrieval should beat
BM25 outright here — the first dataset in this project where that is predicted in
advance, because citation relatedness is not a lexical relation. And if H1 survives
*while* BM25 loses overall, that is a stronger result than either dataset so far has
produced: it would mean the predictor tracks BM25's relative standing even where BM25 is
the wrong tool, which is what a genuine query-level effect should do and what an artefact
of one weak encoder should not.

## 2026-09-21 — Experiment 11 (result): SciDocs refuses the hypothesis

**The pre-registered prediction was wrong.** H1 predicted a positive correlation,
significant at p < 0.05. SciDocs returns rho **−0.0127** (p 0.6810) against MiniLM-L6 and
**+0.0019** (p 0.9546) against bge-small, on 1,000 queries each — the two best-powered
tests this project has run, and both are indistinguishable from zero. H2 fails the same
way: query length gives −0.0113 (p 0.7167) and −0.0199 (p 0.5292), against the −0.22 that
NFCorpus showed.

The terciles say it more plainly than the correlation does. If the effect were present and
merely small, the ordering would survive even where significance did not:

| `max_idf` tercile | n | mean BM25 advantage |
|---|---|---|
| lowest (2.50–5.21) | 334 | −0.0601 |
| middle (5.21–6.77) | 333 | −0.0494 |
| highest (6.80–9.75) | 333 | −0.0665 |

Flat, and not even monotone. On SciFact and NFCorpus the terciles were monotone in all
four cells. There is nothing here to be underpowered about.

### What this does to the project's central claim

The hypothesis has now been asked six times — three datasets by two encoders. Corrected
across that family, which is the honest family because it is one question asked six times:

| Dataset | Encoder | rho | p | Holm (family of 6) |
|---|---|---|---|---|
| SciFact | MiniLM-L6 | +0.1372 | 0.0168 | 0.0674 |
| SciFact | bge-small | +0.1685 | 0.0043 | **0.0255** |
| NFCorpus | MiniLM-L6 | +0.1601 | 0.0042 | **0.0255** |
| NFCorpus | bge-small | +0.0934 | 0.0906 | 0.2719 |
| SciDocs | MiniLM-L6 | −0.0127 | 0.6810 | 1.0000 |
| SciDocs | bge-small | +0.0019 | 0.9546 | 1.0000 |

**Two of six survive correction, and the correlations are no longer all positive.** The
README said "positive in all four, terciles monotone in all four". That is no longer true
of the evidence as a whole, and the claim has to come down to what the measurements
support: the effect is present on two datasets and absent on a third, and it is not a
general property of retrieval over scientific text.

### The mechanism, labelled as the post-hoc story it is

An explanation is available and it should be read with suspicion, because it was
constructed after seeing the result and it conveniently rescues the earlier finding.

SciFact matches a claim against abstracts that state or contradict it; NFCorpus matches a
consumer-health phrase against articles about it. In both, shared terminology is a real
mechanism of relevance, so "the query contains a rare term" is genuinely informative about
whether a lexical matcher will do well. SciDocs asks which papers a given paper *cites*.
Citation is not a lexical relation — a cited paper need not share vocabulary with the
citing title at all — which is why BM25 scores 0.1577 there against 0.6636 on SciFact. If
term rarity predicts BM25's relative standing only where term overlap is a mechanism of
relevance, SciDocs is exactly where it should fail.

That story is consistent with everything measured, and it is also the kind of story that
can be told about any failed replication. The pre-registration said SciDocs was chosen
*because* it was hostile to the hypothesis, which is the one thing that keeps this from
being pure rationalisation: the prediction was still that the effect would show up, and it
was recorded in git before the data was touched. It did not.

**What would actually test it:** a fourth dataset where relevance is topical rather than
citational and BM25 is weak — if the effect returns there, the mechanism story survives; if
it does not, the effect belongs to SciFact and NFCorpus specifically and nothing more.

### The predictions that held

- **H3, the harness check.** Multi-field BM25 scores 0.1577 against BEIR's published
  0.158: a delta of −0.0003, the tightest of the four datasets. The pre-registration noted
  that an absolute ±0.03 tolerance is weak evidence at 0.158; that caveat was correct in
  principle and irrelevant in practice, because the number landed two orders of magnitude
  inside it.
- **Dense beats BM25 outright**, as predicted in advance for the first time in this
  project: MiniLM +0.0587 and bge-small +0.0396 nDCG@10, both Holm p 0.0004, with recall@100
  gaps of +0.1534 and +0.1060.

### An unpredicted result worth more than either

**bge-small lost to MiniLM-L6 on SciDocs** — 0.1973 against 0.2164 — having beaten it
comfortably on both other datasets. It also truncates far less here (2.5% against 30.5%),
so the better-conditioned model on the input side is the worse one on the output side.

Experiment 9 concluded that encoder choice mattered more than anything else measured. That
survives. What does not survive is any implied ranking: "bge-small is the better encoder"
is a statement about SciFact and NFCorpus, not a property of the model. Encoder choice is
worth more than architecture *and* it has to be made per corpus, which is a more expensive
conclusion than the one experiment 9 left standing.

### Fusion, at a `k` fixed in advance

Untuned `k=60` puts RRF below its stronger parent with both encoders: 0.2012 against
MiniLM's 0.2164, and 0.1958 against bge-small's 0.1973 (−0.0015, p 0.7037). On recall@100
fusion is also behind (−0.0086, Holm 0.0495). The pre-registration predicted this
understatement and its cause — the tuned `k` on the other two datasets came out at 1 to 10,
nowhere near 60 — so this is a bound rather than a measurement of what fusion is worth here.
Sweeping `k` on the only split SciDocs has would have produced a better number and no
information.

**Next:** experiment 12 asks whether the surviving two-dataset effect is worth anything in
practice, which after this entry is a question about a smaller thing than it was.

---

## Experiment 12 (pre-registration): is the query-dependence result actionable?

**Written before `run_router.py` was run on anything.**

Experiment 8's result is a correlation, and experiment 9 showed it is a fragile one. But
even a solid correlation is a statement about measurement. The question a reader should
ask, and that this project has not asked of its own central claim, is whether it is worth
anything: if `max_idf` predicts which system will win, route on it and see.

**The method.** For a threshold `t`, queries with `max_idf >= t` are answered by BM25 and
the rest by the dense encoder. Because routing takes one system's entire ranking for a
query, the routed run's score on that query is exactly that system's score on it, so this
needs no retrieval at all — every per-query score is already committed. `t` is swept over
quantiles of `max_idf` on train and scored once on test, like every other parameter here.
The grid includes quantiles 0.0 and 1.0, which degenerate to the two single systems, so
the sweep cannot exclude the baselines the router has to beat.

**The oracle, which is the number that actually matters.** Alongside the real router,
`oracle_assignment` picks the better system per query *by reading the labels*. It is not
a method and cannot be deployed. It is the ceiling: no router driven by any predictor,
present or future, can beat it. That makes it the more informative measurement of the
two, because it answers the general question rather than the specific one.

---

**P1.** The oracle will be far above the better single system — large enough that routing
looks obviously worth doing if you only see the ceiling. Per-query nDCG@10 varies enough
between BM25 and a dense encoder that picking the winner every time should be worth
several points.

**P2.** The `max_idf` router will capture only a small fraction of that headroom, and may
not beat the better single system at all. `max_idf` correlated with the advantage at rho
+0.09 to +0.16; a monotone rule on a predictor that weak cannot recover much of what an
oracle gets from the labels themselves.

**P3.** The router will lose to reciprocal rank fusion. Fusion uses both rankings on every
query; a router commits to one and throws the other away. Routing can only win where
choosing beats combining, and there is no measurement in this project suggesting it does.

**What each outcome would mean.** If P1 holds and P2 holds, the honest summary of this
project's headline finding is that it is real, small, and not currently actionable — the
information is there and `max_idf` is too blunt to extract it. If P1 fails, routing is a
dead end regardless of predictor and the correlation is a curiosity. If P3 fails, that
would be the most interesting result in the repository and would need replicating before
being believed.

## 2026-09-21 — Experiment 12 (result): the effect is real and not worth acting on

**All three pre-registered predictions held.** That is a less comfortable outcome than it
sounds, because what they predicted was failure.

| | BM25 | bge-small | Oracle router | `max_idf` router | vs better single | vs oracle |
|---|---|---|---|---|---|---|
| SciFact | 0.6636 | 0.7200 | **0.7750** | 0.7131 | **−0.0069** | −0.0619 |
| NFCorpus | 0.3253 | 0.3391 | **0.3887** | 0.3425 | +0.0034 | −0.0462 |
| SciDocs | 0.1577 | 0.1973 | **0.2313** | — † | — | — |

† SciDocs has no train split, so no threshold may honestly be chosen; only the ceiling is
reported. It needs no tuning, which is the other reason the oracle is the useful number.

**P1 held: the headroom is large.** Picking the better system per query is worth +0.0550
on SciFact and +0.0496 on NFCorpus over the better system alone. That is more than fusion,
RM3, reranking, stemming or the entire `k1`/`b` space are worth. Seen alone it makes query
routing look like the most valuable unexploited idea in the project.

**P2 held: `max_idf` recovers almost none of it.** The routed run captures +0.0034 of a
possible +0.0496 on NFCorpus — about 7% of the available headroom — and on SciFact it is
*negative*: −0.0069 against simply always using the encoder. A predictor correlating at
rho +0.09 to +0.17 does not support a monotone decision rule, which is the practical
difference between "correlates" and "predicts".

**P3 held: routing loses to fusion.** RRF scores 0.7207 and 0.3610 against the router's
0.7131 and 0.3425, on the same two systems. Fusion uses both rankings on every query; the
router commits to one and discards the other. Nothing measured here suggests choosing ever
beats combining.

### Why the oracle is the number that matters

The oracle reads the relevance labels, so it is a ceiling rather than a method. Its value
is that it bounds *every* router, not just this one, and it needs no training split.

The headroom is large on all three datasets: +0.0550, +0.0496 and +0.0340. That includes
SciDocs, where experiment 11 showed `max_idf` has no predictive power whatsoever
(rho −0.0127 and +0.0019). So the two facts are separable and both are now measured:
**the per-query information exists everywhere, and this predictor finds it nowhere.**

That is a stronger and more useful result than either half alone. It rules out the
comfortable reading of experiment 11 — that SciDocs simply has no query-level structure to
find — and puts the failure squarely on the predictor rather than on the premise.

That separates two claims this project had been treating as one:

- **"Which system wins varies by query."** True everywhere, and the oracle measures it
  directly: 3 to 5.5 points of headroom on all three datasets, including the one where
  `max_idf` predicts nothing.
- **"And term rarity tells you which."** True on SciFact and NFCorpus, false on SciDocs
  (experiment 11), and too weak to act on even where it is true.
- **"Therefore you should route queries."** Not supported anywhere. The correlation is
  real and too weak to act on, and the cheapest thing that does exploit both systems —
  fusing them — beats the router without needing to predict anything.

The third was never measured before this entry, and the first and second had been
collapsed into each other for eleven experiments. That conflation is what made the
headline finding sound more useful than it was: a correlation nobody has tried to use is a
claim with no consequences attached, and separating it from the thing it was standing in
for took one script and no new retrieval.

### What would move this

A better predictor, not a better rule. The oracle says the information is there. `max_idf`
is one number computed from the query and the index before retrieval; a predictor with
access to the retrieval scores themselves — score distribution shape, agreement between
the two rankings, the gap between rank 1 and rank 10 — would have far more to work with,
at the cost of no longer being free. Whether that is worth doing depends on a number this
entry now provides: the ceiling is 5 points, so a predictor recovering even a third of it
would beat fusion. That is the first version of this question with a quantified prize.

---

## Experiment 13 (pre-registration): does an LLM beat 1990s pseudo-relevance feedback?

**Written before any expansion code existed.**

The obvious criticism of this project is that it stops at 2021. Every method measured so
far — BM25, RM3, bi-encoders, a cross-encoder — predates the RAG era, and the gains over
BM25 are modest: +0.057, +0.036 and +0.059 nDCG@10 on SciFact, NFCorpus and SciDocs, with
almost all of it attributable to one decision, using a better encoder.

The cleanest way to test that criticism is not to add a modern method and admire it. It is
to take the modern method whose *mechanism is identical to one already measured here* and
put them against each other on the same harness.

**RM3** reads the top `k` retrieved documents and adds their most distinctive terms to the
query. **HyDE** and **Query2Doc** ask a language model to write the document the query is
looking for, and retrieve with that text appended. Both are query expansion. Both attack
the same failure — the query is short and uses different words from the documents. One
learns the expansion terms from the corpus; the other hallucinates them from parametric
knowledge. This repository already has RM3 measured, tuned on train and scored on test, on
both datasets. That makes it the baseline the newer method has to beat, which is the
ordering this project insists on everywhere else.

**Setup.** `flan-t5-base` generates one pseudo-document per query, locally, offline, with
a fixed seed and greedy decoding so the run reproduces. No API key, no per-run cost, and
the generation is committed so the retrieval can be re-scored without re-generating. The
expanded query is `query * weight + pseudo_document`, with the repetition weight taken
from the Query2Doc paper's formulation; the weight is swept on train and scored once on
test, like every other parameter here.

---

**H1.** HyDE-style expansion beats unexpanded BM25 on NFCorpus, where RM3 already gained
+0.0188 (Holm p 0.0006) and the recall ceiling shows 74% of the achievable documents are
still unretrieved.

**H2.** It does **not** beat unexpanded BM25 on SciFact, where RM3 gained nothing
(−0.0086, p 0.1168) and BM25 has already found 90.1% of what exists. The recall-ceiling
argument is mechanism-level, so it should bind a generative expander exactly as it bound a
statistical one. If HyDE wins on SciFact anyway, that argument is wrong and experiment 4b's
explanation of the RM3 split needs rewriting.

**H3, the one worth running this for.** HyDE does **not** beat RM3 by a Holm-significant
margin on either dataset. A 250M-parameter model writing a plausible abstract is, for
retrieval purposes, doing what relevance-model feedback does — supplying co-occurring
domain vocabulary — and the corpus-grounded version has the advantage of using words that
are actually in the index.

**What each outcome means.** If H3 fails and HyDE wins clearly, the criticism that this
project is dated lands, and the modern stack deserves the rest of the roadmap. If H3
holds, then on these datasets the RAG-era trick reduces to a technique from 2001 with a
GPU attached, and that is worth knowing before anyone builds it into a pipeline.

**Known limitation, stated now rather than as an excuse later.** `flan-t5-base` is small
and not a scientific-domain model. A negative result bounds *this* model, not LLM query
expansion in general — the same bound experiment 7 had to accept for its reranker. The
honest version of H3 is "a small instruction-tuned LM does not beat RM3 here", and the
experiment is designed so that a positive result would be the interesting one.

---

## Experiment 14 (pre-registration): document expansion, the other side of the same gap

**Written at the same time as experiment 13, before either was implemented.**

Experiment 4b explained RM3's split by the recall ceiling: NFCorpus queries are short
consumer-health phrases against clinical writing, so closing the vocabulary gap helps,
while SciFact claims are already written in the register of the abstracts that answer
them. Experiment 13 tests that explanation against a generative expander on the *query*
side. This one tests it from the *document* side, which is where the asymmetry should be
sharpest.

**doc2query** runs a sequence-to-sequence model over every document to predict queries it
would answer, and appends them to the document before indexing. Unlike query expansion it
costs nothing at search time — the expansion is baked into the index — which is the reason
it is worth measuring separately rather than assuming it behaves like RM3.

**Setup.** `doc2query/all-t5-base-msmarco`, five generated queries per document, sampled
with a fixed seed, appended to the `text` field before multi-field indexing. The generated
queries are committed so the index can be rebuilt without re-running generation.

---

**H1.** doc2query helps NFCorpus and not SciFact, mirroring RM3's split, because it
attacks the same vocabulary gap from the other end.

**H2.** The gain on NFCorpus is larger than RM3's +0.0188, because expanding 3,633
documents gives the model far more context per generation than expanding a query does, and
because the expansion is available to every query rather than being re-derived per query
from a noisy top-`k`.

**H3.** doc2query and RM3 are **not** additive on NFCorpus: applying both gains less than
the sum of their individual gains. They are two ways of closing one gap, and a gap can only
be closed once.

**The prediction I am least confident in** is H2, and the reason is worth recording: the
model is trained on MS MARCO, which is web search queries against web passages. Every
MS MARCO-trained model this project has tried has underperformed on scientific text —
the cross-encoder in experiment 7, and the bi-encoders in experiments 5 and 9. If H2 fails
while H1 holds, the consistent reading is that domain mismatch costs more than document
context buys, which would be the fourth independent observation of the same thing.

---

## Experiment 15 (pre-registration): a reranker that was not trained on web search

**Written before the model was downloaded.**

Experiment 7 concluded that cross-encoder reranking bought nothing at the highest cost of
anything measured here, and the migration re-run left that standing on nDCG@10 while
turning up one real gain at rank 1 (+0.0588 on NFCorpus, Holm p 0.0296). Both entries
bound the claim to one model, `ms-marco-MiniLM-L-6-v2`, and say so.

That bound is not a formality. This project has now watched MS MARCO training hurt three
times: the cross-encoder in experiment 7, and the bi-encoders in experiments 5 and 9. MS
MARCO is web-search queries against web passages. Scientific claims and consumer-health
phrases are neither. Leaving "reranking does not work here" attached to the one model most
likely to be mismatched is the same error as experiment 5's — measuring a method with a
badly chosen model and reporting the result as a property of the method.

**Setup.** `BAAI/bge-reranker-base`, reranking the top 100 of the best fused run on each
dataset, against the same fused run unreranked. Same depth, same candidates, same metrics,
same paired test. The only variable is the reranker.

---

**H1.** `bge-reranker-base` beats `ms-marco-MiniLM-L-6-v2` on nDCG@10 on both datasets.
This is the weakest of the three predictions and the one I hold most confidently: it is a
larger, more recent, more broadly trained model.

**H2.** It improves on the *unreranked* fusion on nDCG@10 by a Holm-significant margin on
at least one dataset. Experiment 7's headline conclusion — reranking buys nothing — should
fail once the reranker is not domain-mismatched. If H2 fails, the conclusion generalises
beyond the one model and becomes much stronger than it currently is.

**H3.** The gain, if any, is larger at rank 1 than at rank 10. Reranking reorders a fixed
candidate set, so it cannot add documents; its leverage is concentrated where order matters
most. This is the shape experiment 7 saw on NFCorpus and could not establish, and that the
migration later did establish for the weak model.

**What this cannot settle.** Cost. A reranker roughly twelve times the size of the old one,
over the same candidate depth, is not going to be cheaper, and the interesting comparison
for anyone deploying this is gain per unit of latency rather than gain alone. The timing is
recorded with the result so the trade can be read off, but no claim about it is
pre-registered here.

---

## Experiment 16 (pre-registration): how much does an LLM judge agree with a human?

**Written before the judge was run on anything.**

Roadmap item 13 asked for "answer generation with citations, and an LLM judge calibrated
against human labels". The second half is the measurable one, and it is the half that RAG
evaluation in 2026 almost always skips: a model is asked whether a document is relevant,
or whether an answer is supported, and its verdict is then reported as though it were
ground truth. It is not. It is a measurement made by a model, and nobody quotes its error
rate.

This benchmark has something most RAG evaluations do not: thousands of relevance
judgements made by people. So the judge can be scored the same way any other system here
is scored — against a baseline established first.

**Cohen's kappa, not raw agreement.** Relevance pools are overwhelmingly non-relevant. A
judge that answers "no" to everything scores above 90% raw agreement on a typical pool
while discriminating nothing, and kappa reports that correctly as 0.

**Only two datasets can be used, and finding that out was part of the work.**

| Dataset | Judged pairs | Explicit non-relevant |
|---|---|---|
| SciFact | 339 | 0 |
| NFCorpus | 12,334 | 0 |
| SciDocs | 29,928 | 25,000 |
| TREC-COVID | 66,336 | 41,661 |

SciFact and NFCorpus ship only positive judgements. A judge could be scored there against
the convention that unjudged means non-relevant — which is right for computing nDCG, where
it applies equally to every system, and wrong as ground truth for an assessor, because an
unjudged document is one no human ever looked at and the judge may be correct about it.
`judgeable_pairs` raises rather than quietly returning a single-class pool.

**Setup.** `flan-t5-base` is asked, for each judged document in the top 10 of the best
retrieval run, whether it is relevant to the query. Verdicts are parsed to yes/no;
anything else is counted as unparseable and dropped rather than coerced, because coercing
to "no" on a mostly-non-relevant pool would inflate agreement.

---

**H1.** Kappa is **below 0.4** on both datasets — fair agreement at best by the
conventional reading, and well short of what would justify substituting the judge for a
human assessor. A 250M-parameter instruction-tuned model is not a trained relevance
assessor and has no access to the assessment guidelines the humans worked from.

**H2.** The judge is biased toward **yes**. Its errors are mostly non-relevant documents
called relevant, not the reverse. The pairs it sees are the top 10 of a retrieval run, so
every one of them is topically plausible, and the question "is this relevant" is much
easier to answer affirmatively than the humans' actual question, which is whether the
document addresses the specific information need.

**H3.** Kappa is **higher on TREC-COVID than on SciDocs.** TREC-COVID relevance is topical
— does this paper concern this aspect of COVID-19 — which is close to what an instruction-
tuned model can assess. SciDocs relevance is citational: whether one paper cites another
is not something the text of either reveals, and experiment 11 already showed that
lexical signal predicts nothing there.

**Why H3 is the interesting one.** If it holds, the reading is that LLM judges work on
tasks where relevance is semantic and fail where it is relational — which is a constraint
on where LLM-as-judge can be used at all, not a statement about this particular model. If
kappa is near zero on both, the honest conclusion is stronger and simpler: an
uncalibrated LLM judge is not evidence, and any RAG evaluation resting on one is
reporting the model's prior rather than a measurement.

**What this does not test.** Answer faithfulness, which is the other thing LLM judges are
used for, and a larger judge. A negative result here bounds `flan-t5-base` on relevance
assessment, exactly as experiment 7's bounded its reranker; the difference is that this
time the bound is stated before the run rather than after.

---

## 2026-09-21 — Experiment 16 (result): what an uncalibrated LLM judge is worth

**The headline is the gap between two numbers that describe the same run.**

On SciDocs, over 4,000 human-labelled pairs, `flan-t5-base` agrees with the human
assessors on **74.9%** of them. Reported alone, that reads as a serviceable judge. Chance
agreement, given how often each side says "relevant", is **70.7%**. Cohen's kappa is
therefore **0.1433** — slight agreement, and nearly all of the raw figure is the two
parties independently saying "no" to a pool that is mostly non-relevant.

TREC-COVID is the same story with different arithmetic: 66.5% raw, 58.3% chance, kappa
0.1965. Anyone reporting the raw figure has overstated their judge by roughly a factor of
four on one dataset and three on the other.

| | SciDocs | TREC-COVID |
|---|---|---|
| Pairs | 4,000 | 4,000 |
| Human called relevant | 606 (15.2%) | 1,489 (37.2%) |
| Judge called relevant | 812 (20.3%) | 709 (17.7%) |
| Unparseable verdicts | 0 | 0 |
| Raw agreement | 0.7490 | 0.6645 |
| Chance agreement | 0.7070 | 0.5825 |
| **Cohen's kappa** | **0.1433** | **0.1965** |
| Called relevant, human did not | 605 | 281 |
| Missed a relevant one | 399 | 1,061 |

**H1 held on both.** Predicted kappa below 0.4; measured 0.1433 and 0.1965.

**H3 held.** Predicted kappa higher on TREC-COVID than SciDocs, because topical relevance
is something an instruction-tuned model can assess and citation relevance is not something
either document's text reveals. 0.1965 against 0.1433.

**H2 failed, and the failure is worth more than the prediction was.** I predicted the
judge would lean toward yes, on the reasoning that every pair it sees is topically
plausible. That holds on SciDocs — 20.3% against the humans' 15.2% — and inverts on
TREC-COVID, where it calls 17.7% relevant against a human rate of 37.2% and misses 1,061
relevant documents.

Putting the two rows together explains both. **The judge says yes on 17.7% and 20.3% of
pairs, on datasets whose true relevance rates are 37.2% and 15.2%.** Its positive rate is
near-constant while the underlying prevalence differs by a factor of 2.4. It is not biased
toward yes or toward no; it has a fixed answering rate that does not track the data at all,
and which direction it *appears* biased in is a property of the dataset rather than of the
judge.

That is a more useful characterisation than H2's, and it is also more damaging. A judge
with a fixed positive rate cannot be corrected by moving a threshold, because the thing it
is failing to do is respond to prevalence. And an evaluator comparing two systems on a
corpus where relevance is dense will be told, by this judge, that both retrieve far less
than they do.

### The confusion matrix matters more than kappa

On SciDocs the judge found 207 of 606 relevant documents, and 605 of the 812 it called
relevant were not. On TREC-COVID it found 428 of 1,489. Neither is a conservative judge or
an eager one; both are weakly correlated judges, wrong in both directions at once. Kappa
compresses that into one number, which is why the counts are recorded beside it.

### The methodological failure that came first, and is the more useful lesson

The first run of this experiment returned kappa 0.0109 and it meant nothing at all.

The pool was sampled from the top 10 of a BM25 run, on the reasoning that those are the
documents a RAG system would actually put in front of a model. On SciDocs that produced
819 labelled pairs of which **814 were relevant**. The corpus is large, the judged pool is
small relative to it, and the non-relevant judgements are hard negatives that a lexical
matcher does not surface — so the "realistic" pool was 99.4% one class. With a pool that
skewed, chance agreement approaches the observed rate and kappa approaches zero regardless
of how good the judge is.

That run is preserved as `results/scidocs-judge-retrievedpool.json`, reproducible with
`--source retrieved --min-per-class 1`, which is the only reason the balance guard is
adjustable at all. It was deleted when the experiment was re-run and had to be
regenerated, because this log quotes its numbers and a log entry describing a run nothing
produced is the failure the documentation test exists to catch. It came back
bit-identical.

The guard written specifically to prevent this checked the wrong object. It verified that
the *dataset* contained non-relevant judgements, which SciDocs does, 25,000 of them. It
never checked that the *sample* did. The check now runs on the pool that is actually
returned and refuses fewer than thirty of either class.

That is the transferable part of this entry. A skew guard on the population is not a skew
guard on the sample, and an agreement statistic computed over a degenerate sample fails
quietly — it returns a plausible small number rather than an error.


### What this means for evaluating RAG

Neither kappa clears 0.2. On the conventional reading that is "slight" agreement, and it
is the level at which a judge's verdict should be treated as a noisy signal rather than a
label. An evaluation that substitutes this judge for a human assessor is not measuring
retrieval quality; it is measuring a model's prior, with an error rate four times larger
than its raw agreement suggests.

That bounds `flan-t5-base` and nothing more — the same bound experiment 7 had to accept
for its reranker, and stated in advance here rather than after the fact. A larger judge
would very likely do better. The transferable claim is not "LLM judges do not work", it is
that **an uncalibrated one is not evidence**, and that the cost of calibrating is one
afternoon against a benchmark that ships human labels.

**Next:** the obvious follow-up is a larger judge on the same pools, which would turn this
from one point into a curve. The infrastructure is now in place for that to be a single
command.

---

## In progress: experiments 13, 14 and 15

All three are pre-registered above. Generation is the long pole and runs detached, so
this section exists to survive a context reset — everything needed to finish them is here
or in the scripts.

**State as of writing.** Query expansion generation is on NFCorpus train (2590 queries at
about 1.6s each). SciFact test and train follow. Document generation starts after that,
via a script that stops the query queue first.

**Document generation runs at the pre-registered five queries per document.** An earlier
version of that script used three, to save wall clock. That was a shortcut rather than a
measured decision, and it was the wrong one: experiment 14's H2 predicts a gain *larger*
than RM3's, so generating less expansion text biases the test in the direction that makes
its own hypothesis fail. All three of experiment 14's hypotheses bind on NFCorpus, so
NFCorpus generates first as well as at full strength. Overnight is an acceptable price for
measuring what was actually predicted.

**What fires automatically.** `scratchpad/pipeline.sh` waits on file markers, not process
names, and runs:

- experiment 13 when `data/expansions/scifact-query-train.json` appears — sweeps the
  query weight on train for both datasets, scores on test, then compares HyDE against
  BM25 *and* against RM3;
- experiment 15 immediately after, since it needs no generation;
- experiment 14 when `data/expansions/scifact-document-test.json` appears.

Results land in `results/` as `*-query-expansion-hyde.json`,
`*-document-expansion-doc2query.json`, `*-rrf-rerank-bgererank.json` and the paired
comparisons beside them.

Re-run checklist:

- [ ] Experiment 13 entry written, against H1, H2 and H3 as pre-registered
- [ ] Experiment 14 entry written, including H3 (doc2query and RM3 should not stack)
- [ ] Experiment 15 entry written, and experiment 7's bounded conclusion revisited
- [ ] README narrative and Annexe A updated with whichever of these produced a result
- [ ] Roadmap items 13 to 15 marked done rather than "running"
- [ ] `docs/decisions.md` updated if any default changed

**Three process notes, all learned the hard way tonight.** Do not poll with
`pgrep -f <name>` when the waiter's own command line contains that name: the waiters match
each other and deadlock, which cost about twenty minutes twice. Use file markers. Redirect
stderr when running generation in the background, or a tqdm progress bar floods the log.
And when two detached scripts both react to the same marker file — here, one queue
finishing its queries and another starting its documents — the slower poller loses: both
launch a language model, and two torch processes on this machine drive it into swap. The
waiter that should win polls every two seconds and keeps stopping the other for a minute
before starting any work of its own.

**If a result file is deleted or overwritten**, regenerate it before quoting its numbers.
`results/scidocs-judge-retrievedpool.json` had to be regenerated for exactly this reason,
and the documentation test is what caught it.

---

## Still open

| Question | Settles |
|---|---|
| ~~Domain-matched encoder~~ | **Done (experiment 9)** — general model quality beat domain matching; experiment 5's conclusion overturned |
| ~~Chunking instead of truncation~~ | **Done (experiment 9)** — removing truncation made results slightly worse; the tails carried no signal |
| ~~TREC-COVID query formulation~~ | **Done (experiment 10)** — not the cause; multi-field indexing was |
| ~~Migrate to multi-field BM25, or not~~ | **Done** — migrated, everything re-run, three conclusions moved |
| ~~Query-length effect, pre-registered~~ | **Done (experiment 11)** — pre-registered on SciDocs and failed: rho −0.0113 and −0.0199. The NFCorpus lead does not generalise |
| **A predictor that recovers the routing headroom** | Experiment 12 measured a 3–5.5 point ceiling on every dataset and showed `max_idf` finds none of it. A predictor using retrieval-time signal — score distribution shape, agreement between rankings, the rank-1-to-rank-10 gap — has far more to work with. It needs to beat fusion, not just BM25 |
| **A fourth dataset, topical relevance, weak BM25** | Experiment 11's post-hoc story is that term rarity predicts BM25's standing only where term overlap is a mechanism of relevance. SciDocs is citational *and* BM25-hostile, so it cannot separate those two explanations. A topical dataset where BM25 is weak would |
| Domain-matched cross-encoder | Experiment 12's reranking result bounds `ms-marco-MiniLM-L-6-v2` only, and the one place it did help — nDCG@1 on NFCorpus, Holm 0.0296 — suggests the top of the ranking is where to look |
| Paired tests on TREC-COVID | Only 50 queries, so almost nothing will be detectable; worth confirming that explicitly rather than leaving it implied |

## Notes to self

- Graded qrels (TREC-COVID, NFCorpus) make the exponential-vs-linear gain choice
  matter. Record which was used in every entry from experiment 4 onward.
- Before claiming a hybrid win, check it survives a paired significance test. On 300
  queries, differences under about a point are noise.
- SciFact is a weak showcase for hybrid retrieval because BM25 is already strong on
  it. Expect thin gains and do not over-read them; TREC-COVID has the headroom.
