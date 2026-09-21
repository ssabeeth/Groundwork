# Groundwork

Retrieval over scientific literature, with the numbers measured rather than assumed.

Most RAG projects pick a retrieval method and move on. This one treats the choice as the
experiment: BM25, RM3, dense, hybrid, reranked, expanded and routed retrieval are scored
against the same human relevance judgements, with a tested evaluation harness, and the
results are reported including the ones that did not help — and including the ones this
project got wrong and had to take back.

The short version of what it found: **what a model was trained to optimise decides
whether it helps — not how modern it is.**

The methods trained to *generate plausible text* all failed. LLM query expansion loses to
RM3, a pseudo-relevance-feedback technique from 2001, by a Holm-significant margin.
doc2query expands every document in a corpus — a median of six new indexable terms each —
and retrieval does not move. A modern cross-encoder reranks *worse* than the small one it
was meant to replace, and on nDCG@10 both leave a good ranking worse than they found it.

The methods trained *against relevance* did better. **SPLADE**, which learns sparse term
weights end to end, beats BM25 on both datasets and beats both generative expanders by
0.0341 and 0.0287 (Holm 0.0006) — it expands too, so the difference is what the expansion
was optimised for. **ColBERT** reproduces but loses to a bi-encoder a fraction of its size.

Through all of it the most boring option kept winning: fusing a lexical and a dense
retriever beats both of its own components at Holm 0.0006 or below, and no single
retriever measured here — including SPLADE — has beaten it. No router built on this
project's own central finding beat simply doing that either.

**Status:** eighteen experiments across four BEIR datasets, all complete, with an LLM judge,
cited answer generation and an MCP server on top. Every method is measured on one harness
with paired significance testing throughout, and every figure in this file is checked
against a committed run by CI.

Start with the note below. Eight experiments were pre-registered in the log and committed
to git before their data was touched. **In six of those eight, at least one prediction
turned out wrong** — including the one this project's central claim rested on — and every
one of them is still on the page. Three further results replaced earlier results of this project's
own.

---

## Note: what this project got wrong, and what changed

Read this before the numbers.

Three things below are corrections this project made to itself. They are at the top rather
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

**An uncalibrated LLM judge is not evidence.** Asked whether a retrieved document is
relevant, `flan-t5-base` agrees with the human assessors 74.9% of the time on SciDocs —
which sounds usable until you subtract the 70.7% two parties reach by chance on a pool
that is mostly non-relevant. Cohen's kappa is 0.1433, and 0.1965 on TREC-COVID. Reporting
raw agreement overstates the judge by three to four times. This is the standard way RAG
systems are evaluated in 2026 and it is almost never calibrated, which is possible here
only because the benchmark ships thousands of human decisions to check against.

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
## What did work

**Learned sparse retrieval.** SPLADE learns a weight for every term in the BERT vocabulary
and scores with a dot product over an inverted index — the same operation as BM25, so the
cost structure is the same too: indexing is a one-off and retrieval is seconds. It beats
BM25 by **+0.0443 on SciFact (Holm 0.0069) and +0.0295 on NFCorpus (Holm 0.0006)**, and it
reproduces its published BEIR figures to +0.0149 and +0.0068.

It is also the experiment that corrected two earlier ones. Experiments 13 and 14 found that
LLM query expansion and doc2query both failed, which reads like "expansion does not help on
this data". SPLADE expands as well — a document gains weight on terms it does not contain —
and beats HyDE by 0.0341 and doc2query by 0.0287, both at Holm 0.0006. The difference is
not care, it is objective: those two were trained to produce text a human would judge
plausible, and SPLADE was trained against relevance judgements. Only one of those is the
thing being measured.

One claim this repository made four times is now dead because of it. Every time an MS
MARCO-trained model underperformed here — a cross-encoder, two bi-encoders, doc2query — the
explanation offered was domain mismatch. **SPLADE is MS MARCO-trained and is the best
single retriever measured here**, so that explanation is refuted as stated. Something more
specific is wrong in those four cases, and nothing here identifies what.

**Fusion, still.** SPLADE does not beat the BM25+dense fusion on either dataset (−0.0128
and −0.0062). Six methods in, no single retriever has.

## What didn't work

**LLM query expansion loses to a technique from 2001 — and mostly never ran.** HyDE and
Query2Doc ask a language model to write the document a query is looking for and retrieve
with that text appended. RM3, already measured here, closes the same vocabulary gap by
reading the corpus instead. Put against each other on the same harness, with the
expansion weight tuned on train and scored once on test, **RM3 wins on NFCorpus by 0.0234
nDCG@10, Holm p 0.0004.** Against plain BM25 the expansion is −0.0046 on NFCorpus and
+0.0007 on SciFact, neither detectable.

The interesting part is why, and it was measured before the scoring rather than argued
afterwards. `scripts/analyse_expansions.py` counts the *new indexable terms* an expansion
adds — terms absent from the query after stemming and stopword removal, since only those
change which documents can match. On SciFact **249 of 300 expansions add none at all**,
and 0.0349 of the generated vocabulary is new: `flan-t5-base`, asked to write an abstract
answering a claim, restates the claim. RM3 adds 50 terms. The paired test agrees from the
other direction, for free — its tie count says the expansion changed nDCG@10 for **7
SciFact queries out of 300**, and recall@100 for 1.

So this is not evidence that HyDE does not work. It bounds one small model, and the
vocabulary count says that model largely did not perform the technique; a larger one would
add more terms and might well win. What transfers is the check rather than the verdict:
**before believing a generative expansion helped, count the new terms it added.** That
number is free, available before any retrieval runs, and on SciFact it predicts the null
result without scoring a single ranking.

**Document expansion expanded, and nothing moved.** doc2query generates five queries per
document and appends them before indexing — 3,633 and 5,183 documents, all of them. Unlike
the query expander above, it genuinely did the job: **9 of 3,633 NFCorpus documents gained
no new indexable term**, and the median document gained six. Retrieval barely noticed.
Against BM25 it is +0.0008 on NFCorpus (Holm 1.0000) and +0.0059 on SciFact (Holm 0.6047),
and against RM3 on NFCorpus it is **worse by 0.0180, Holm 0.0050**.

The pre-registered prediction that it would mirror RM3's dataset split failed, and took an
earlier explanation with it. Experiment 4b attributed RM3's split to a recall ceiling —
SciFact claims already read like the abstracts that answer them, so closing the vocabulary
gap cannot help there. The gap was closed from the document side for every document, and
SciFact is where the point estimate is larger. That explanation does not generalise.

The prediction that doc2query and RM3 would not stack also failed, in the opposite
direction: RM3 gains **more** over an already-expanded corpus (+0.0247, Holm 0.0006) than
over a raw one (+0.0188). Best guess, offered as the post-hoc story it is: doc2query does
not help retrieval directly, it improves the documents RM3 reads its feedback terms from.
The one cell where it measurably pays was found after the fact and is flagged as post-hoc
in the log — adding it to RM3 on SciFact is +0.0145 (Holm 0.0170), which is the difference
between RM3 hurting that dataset and not.

One more thing worth flagging about tuning. The NFCorpus sweep chose weight 20, the
largest on its grid, with nDCG@10 rising monotonically across all eight cells. The weight
is how many times the *original* query is repeated, so a larger weight dilutes the
generated text: the tuner was asking for less expansion, and the limit of that direction
is not a better setting of the method but unexpanded BM25. "Best weight 20" reads like a
tuned parameter and is not one, so the sweep now records `best_at_grid_edge` and
`monotone` alongside the peak.

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

That bound is now gone, and the conclusion is stronger for it. **A better reranker
reranks worse.** `BAAI/bge-reranker-base` — larger, more recent, not trained on MS MARCO —
was pre-registered with the prediction that it would beat both MiniLM and the unreranked
fusion. It does neither. On NFCorpus it takes the best fused run from 0.3610 down to
0.3155 (Holm 0.0004), below plain BM25 at 0.3253, and it is Holm-significantly *worse*
than the small MS MARCO model it was supposed to beat (−0.0402). On SciFact it is also
below the unreranked fusion, 0.7095 against 0.7207.

So the explanation is not domain mismatch. Two cross-encoders with different training
corpora and an order of magnitude between them both make a good fusion worse on this data.
That is also the opposite of what happened with bi-encoders, where general model quality
did transfer — whatever makes a good bi-encoder here does not make a good reranker here.

The cost is recorded with the result: **1220 and 1292 seconds** to rescore 300 and 323
queries × 100 candidates, against roughly fifteen seconds to build the fusion being
degraded. Twenty minutes, per dataset, to lose accuracy.

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

## What was built and measured

All complete. Numbered as in [`docs/experiments.md`](docs/experiments.md), which is the
full chronological log, including the runs that settled nothing.

| # | Experiment | What it settled |
|---|---|---|
| 1 | Tested metrics and BM25 baseline | Four published BEIR baselines reproduced, three within 0.0014 |
| 2 | Tokenisation ablation | Stemming helps ranking: +0.0309 nDCG@10, Holm 0.0156 |
| 3 | `k1`/`b` sweep, tuned on train | Tuning buys nothing on test: +0.0000, p 0.9899 |
| 4 | BM25 on graded qrels | Exponential vs linear gain, on NFCorpus and TREC-COVID |
| 4b | RM3 pseudo-relevance feedback | Helps NFCorpus (+0.0188), hurts SciFact (−0.0086) |
| 5 | Dense retrieval on the same harness | Claimed dense never beats BM25 — **later overturned by 9** |
| 6 | Hybrid by reciprocal rank fusion | Beats both its components on both datasets, Holm ≤ 0.0006 |
| 7 | Cross-encoder reranking | Costs the most of anything here and earns almost nothing |
| 8 | Results broken down by query type | The project's central finding: which method wins varies by query |
| 9 | Encoder choice and chunking | Falsification test of 8. Overturned 5; halved 8 |
| 10 | Multi-field indexing | Nine experiments had been indexed wrongly. Everything re-run; two conclusions reversed |
| 11 | SciDocs, pre-registered | **Prediction failed.** The query-length effect does not generalise |
| 12 | Query routing | 3–5.5 points of headroom exist; no router recovers any of it. Fusion wins |
| 13 | LLM query expansion vs RM3 | **RM3 from 2001 wins**, Holm 0.0004. The expander mostly restated the query |
| 14 | Document expansion with doc2query | Expanded every document, moved nothing. All three predictions failed |
| 15 | A reranker not trained on web search | Reranks *worse* than the small model it replaced. All three predictions failed |
| 16 | An LLM judge vs human assessors | Cohen's kappa 0.1433 and 0.1965 — slight agreement, not ground truth |
| 17 | Answer generation with citations | Citations checked against the retrieved set; answer quality declared unmeasurable here |
| 18 | MCP server | Exposes the measured comparison, not a hidden "best" method |
| 19 | Learned sparse retrieval (SPLADE) | **All three predictions held.** Beats BM25, RM3 and both generative expanders; still below fusion |
| 20 | Late interaction (ColBERT) | One of three held. Loses to a bi-encoder a fraction of its size; two silent implementation bugs found by the reproduction check |

Experiments 13 to 16 exist because "this project stops at 2021" is a fair criticism. Each
puts a method from the RAG era against the older method with the same mechanism, measured
on the same harness — rather than adding a modern method and admiring it.

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

### A.9 What an LLM judge is worth

Both datasets whose qrels carry explicit non-relevant judgements, 4,000 human-labelled
pairs each, judged by `flan-t5-base`:

| | SciDocs | TREC-COVID |
|---|---|---|
| Human called relevant | 15.2% | 37.2% |
| Judge called relevant | 20.3% | 17.7% |
| Raw agreement | 0.7490 | 0.6645 |
| Chance agreement | 0.7070 | 0.5825 |
| **Cohen's kappa** | **0.1433** | **0.1965** |

```bash
python scripts/run_judge.py --dataset scidocs --max-pairs 4000
```

Kappa rather than raw agreement, because relevance pools are mostly non-relevant and a
judge answering "no" to everything scores above 90% raw while discriminating nothing.

The two middle rows are the result. The judge says yes on 17.7% and 20.3% of pairs while
the true rates are 37.2% and 15.2% — a near-constant answering rate against a prevalence
that differs by a factor of 2.4. It is not biased toward yes or no; it does not track
prevalence at all, so which way it *appears* biased is a property of the dataset. A
threshold cannot fix that.

**Only two of the four datasets can be used**, and finding that out was part of the work.
SciFact and NFCorpus ship no non-relevant judgements — 339 and 12,334 judged pairs, every
one relevant. A judge could be scored there against the convention that unjudged means
non-relevant, which is right for computing nDCG and wrong as ground truth for an assessor,
because an unjudged document is one no human looked at. The code refuses those datasets.

### A.10 LLM query expansion against RM3

`flan-t5-base` writes one pseudo-document per query; the expanded query is the original
repeated `weight` times followed by that passage. Weight swept on train, scored once on
test. Tuned weights: 20 on NFCorpus, 8 on SciFact.

| | NFCorpus BM25 | NFCorpus RM3 | NFCorpus HyDE | SciFact BM25 | SciFact RM3 | SciFact HyDE |
|---|---|---|---|---|---|---|
| nDCG@10 | 0.3253 | **0.3440** | 0.3207 | **0.6636** | 0.6550 | 0.6643 |
| Recall@100 | 0.2494 | **0.3121** | 0.2560 | 0.9009 | **0.9053** | 0.9020 |

Paired randomisation, Holm-corrected across the family of four:

| Comparison | nDCG@10 | Holm | Recall@100 | Holm |
|---|---|---|---|---|
| NFCorpus HyDE − BM25 | −0.0046 | 0.2553 | +0.0066 | 0.0165 |
| NFCorpus HyDE − RM3 | **−0.0234** | **0.0004** | **−0.0561** | **0.0004** |
| SciFact HyDE − BM25 | +0.0007 | 0.5426 | +0.0011 | 1.0000 |
| SciFact HyDE − RM3 | +0.0093 | 0.2553 | −0.0033 | 1.0000 |

```bash
python scripts/generate_expansions.py --dataset nfcorpus --kind query --split train
python scripts/run_expanded.py --dataset nfcorpus --kind query --split train --sweep
python scripts/run_expanded.py --dataset nfcorpus --kind query --weight 20 --tag hyde
```

**What the expansions contain**, measured before any of the above was scored. New
indexable terms are those absent from the original query after stemming and stopword
removal — the only ones that can change which documents match:

| | NFCorpus (test) | SciFact (test) |
|---|---|---|
| Queries | 323 | 300 |
| Expansions adding no new term | 99 | **249** |
| Median new terms per query | 2 | **0** |
| New fraction of generated vocabulary | 0.4498 | **0.0349** |
| Terms RM3 adds, for comparison | 50 | 50 |

```bash
python scripts/analyse_expansions.py --dataset scifact --kind query --split test
```

The paired test reaches the same conclusion independently, through the queries on which
the two runs score identically: against BM25 on SciFact the expansion moved nDCG@10 for
7 queries and recall@100 for 1.

Two tuning details recorded with the sweeps. NFCorpus peaked at the edge of its grid
(`best_at_grid_edge`, curve `increasing`, span 0.0795), which for this parameter means the
tuner asking for less expansion rather than a located optimum; SciFact's surface is flat
(span 0.0183) and peaks in the interior.

### A.11 Two rerankers over the same candidates

`BAAI/bge-reranker-base` against experiment 7's `ms-marco-MiniLM-L-6-v2`, both rescoring
the top 100 of the best fused run on each dataset. Only the reranker changes.

| SciFact (RRF k=1.0) | fusion | MiniLM | bge-reranker |
|---|---|---|---|
| nDCG@1 | 0.5900 | 0.5800 | 0.5900 |
| nDCG@10 | **0.7207** | 0.6881 | 0.7095 |
| nDCG@100 | 0.7443 | 0.7216 | 0.7368 |
| Recall@100 | 0.9567 | 0.9567 | 0.9567 |

| NFCorpus (RRF k=5.0) | fusion | MiniLM | bge-reranker |
|---|---|---|---|
| nDCG@1 | 0.4572 | **0.4840** | 0.4314 |
| nDCG@10 | **0.3610** | 0.3557 | 0.3155 |
| nDCG@100 | 0.3336 | 0.3325 | 0.3081 |
| Recall@100 | 0.3149 | 0.3149 | 0.3149 |

Paired randomisation, Holm-corrected across the family of four:

| Comparison | nDCG@10 | Holm | nDCG@1 | Holm |
|---|---|---|---|---|
| SciFact bge-reranker − fusion | −0.0112 | 0.4111 | +0.0000 | 1.0000 |
| NFCorpus bge-reranker − fusion | **−0.0455** | **0.0004** | −0.0258 | 0.8768 |
| SciFact bge-reranker − MiniLM | +0.0214 | 0.2682 | +0.0100 | 1.0000 |
| NFCorpus bge-reranker − MiniLM | **−0.0402** | **0.0004** | **−0.0526** | **0.0448** |

```bash
python scripts/run_hybrid.py --dataset nfcorpus --k 5.0 --model BAAI/bge-small-en-v1.5 \
    --rerank --rerank-model BAAI/bge-reranker-base --tag bgererank
```

Recall@100 is identical down every column because reranking reorders a fixed candidate set
and cannot add to it. That row is the control.

Cost, recorded with each run:

| | pairs rescored | rerank seconds | fusion build seconds |
|---|---|---|---|
| SciFact | 300 × 100 | 1220.1 | 18.98 |
| NFCorpus | 323 × 100 | 1292.4 | 14.45 |

### A.12 Document expansion with doc2query

`castorini/doc2query-t5-base-msmarco`, five sampled queries per document, appended to the
`text` field before multi-field indexing.

| NFCorpus | BM25 | doc2query | RM3 | RM3 over expanded |
|---|---|---|---|---|
| nDCG@10 | 0.3253 | 0.3260 | 0.3440 | **0.3507** |
| Recall@100 | 0.2494 | 0.2515 | 0.3121 | **0.3170** |

| SciFact | BM25 | doc2query | RM3 | RM3 over expanded |
|---|---|---|---|---|
| nDCG@10 | 0.6636 | **0.6695** | 0.6550 | **0.6695** |
| Recall@100 | 0.9009 | 0.9020 | 0.9053 | **0.9087** |

Paired randomisation, Holm-corrected across the pre-registered family of six:

| Comparison | nDCG@10 | Holm | Recall@100 | Holm |
|---|---|---|---|---|
| NFCorpus doc2query − BM25 | +0.0008 | 1.0000 | +0.0021 | 0.9103 |
| NFCorpus doc2query − RM3 | **−0.0180** | **0.0050** | **−0.0606** | **0.0006** |
| NFCorpus (RM3 over expanded) − doc2query | **+0.0247** | **0.0006** | **+0.0655** | **0.0006** |
| SciFact doc2query − BM25 | +0.0059 | 0.6047 | +0.0011 | 1.0000 |
| SciFact doc2query − RM3 | +0.0145 | 0.1068 | −0.0033 | 1.0000 |
| SciFact (RM3 over expanded) − doc2query | +0.0000 | 1.0000 | +0.0067 | 1.0000 |

```bash
python scripts/generate_expansions.py --dataset nfcorpus --kind document --batch-size 8
python scripts/run_expanded.py --dataset nfcorpus --kind document --tag doc2query
python scripts/run_rm3.py --dataset nfcorpus --fb-docs 5 --fb-terms 50 --alpha 0.8 \
    --document-expansions data/expansions/nfcorpus-document-test.json --tag doc2query-rm3
```

**What the expansions contain.** Unlike the query expander in A.10, this one did the job:

| | documents | adding no new term | median new terms | new fraction of generated vocabulary |
|---|---|---|---|---|
| NFCorpus | 3633 | 9 | 6 | 0.4133 |
| SciFact | 5183 | 11 | 5 | 0.3697 |

**Post-hoc, not pre-registered** — whether doc2query adds anything on top of RM3, Holm
corrected within its own family of two:

| | nDCG@10 | Holm | Recall@100 | Holm |
|---|---|---|---|---|
| NFCorpus RM3+doc2query − RM3 | +0.0067 | 0.0692 | +0.0049 | 0.1610 |
| SciFact RM3+doc2query − RM3 | **+0.0145** | **0.0170** | +0.0033 | 1.0000 |

### A.13 Learned sparse and late interaction

`naver/splade-cocondenser-ensembledistil` and `colbert-ir/colbertv2.0`, both with documents
capped at 512 tokens, scored on the same harness as everything else.

| nDCG@10 | SciFact | NFCorpus |
|---|---|---|
| BM25 | 0.6636 | 0.3253 |
| RM3 | 0.6550 | 0.3440 |
| HyDE | 0.6643 | 0.3207 |
| doc2query | 0.6695 | 0.3260 |
| ColBERT | 0.6959 | 0.3535 |
| **SPLADE** | **0.7079** | **0.3548** |
| bge-small dense | 0.7200 | 0.3391 |
| RRF fusion, BM25+dense | **0.7207** | **0.3610** |
| RRF fusion, BM25+ColBERT | 0.7053 | 0.3488 |

Reproduction against published BEIR figures, which is the check everything else rests on:

| | measured | published | delta |
|---|---|---|---|
| SPLADE, SciFact | 0.7079 | 0.693 | +0.0149 |
| SPLADE, NFCorpus | 0.3548 | 0.348 | +0.0068 |
| ColBERT, SciFact | 0.6959 | 0.693 | +0.0029 |
| ColBERT, NFCorpus | 0.3535 | 0.338 | +0.0155 |

Paired randomisation, Holm-corrected within each family of six:

| Comparison | nDCG@10 | Holm | Recall@100 | Holm |
|---|---|---|---|---|
| SPLADE − BM25, SciFact | **+0.0443** | **0.0069** | **+0.0478** | **0.0063** |
| SPLADE − BM25, NFCorpus | **+0.0295** | **0.0006** | **+0.0397** | **0.0006** |
| SPLADE − fusion, SciFact | −0.0128 | 0.7021 | −0.0080 | 0.5090 |
| SPLADE − fusion, NFCorpus | −0.0062 | 0.7021 | **−0.0257** | **0.0063** |
| SPLADE − HyDE, NFCorpus | **+0.0341** | **0.0006** | **+0.0331** | **0.0006** |
| SPLADE − doc2query, NFCorpus | **+0.0287** | **0.0006** | **+0.0376** | **0.0006** |
| ColBERT − BM25, SciFact | +0.0322 | 0.1515 | +0.0278 | 0.2148 |
| ColBERT − BM25, NFCorpus | **+0.0282** | **0.0006** | **+0.0362** | **0.0006** |
| ColBERT − bge-small, SciFact | −0.0241 | 0.4692 | −0.0247 | 0.3096 |
| ColBERT − bge-small, NFCorpus | +0.0144 | 0.4692 | −0.0204 | 0.0865 |
| ColBERT fusion − ColBERT, SciFact | +0.0095 | 0.5383 | +0.0067 | 1.0000 |
| ColBERT fusion − ColBERT, NFCorpus | −0.0046 | 0.5383 | −0.0002 | 1.0000 |

```bash
python scripts/run_learned.py --dataset scifact --method splade --max-length 512
python scripts/run_learned.py --dataset scifact --method colbert --doc-length 512
python scripts/run_hybrid.py --dataset scifact --systems bm25,colbert --k 5.0
```

**Truncation is not a detail.** Both methods' defaults truncate these corpora heavily, and
the cost differs by a factor of ten:

| | short | long | delta nDCG@10 | Holm |
|---|---|---|---|---|
| SPLADE, SciFact | 256 | 512 | +0.0055 | 0.5672 |
| SPLADE, NFCorpus | 256 | 512 | +0.0049 | 0.2342 |
| ColBERT, SciFact | 180 | 512 | **+0.0494** | **0.0004** |
| ColBERT, SciFact | 180 | 300 | **+0.0453** | **0.0006** |

SPLADE takes a maximum over positions, so a term mentioned anywhere early carries the
document. ColBERT matches query tokens against document tokens, so a truncated document
loses those matches outright. With experiment 9 — where chunking a *bi-encoder* to see
whole documents made results slightly worse — that is three representation types answering
the same question differently.

Cost, recorded with each run:

| | index | retrieve |
|---|---|---|
| SPLADE, SciFact | 292.1s | 7.4s |
| ColBERT, SciFact | 213.8s | 35.4s |

ColBERT here is exhaustive MaxSim with none of its serving machinery — no centroid
candidate generation, no residual compression, no PLAID — so these bound quality, not
efficiency, and no efficiency claim is made from them.

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
