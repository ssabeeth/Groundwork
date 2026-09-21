# Design decisions

Short notes on choices that are not obvious from the code, so the reasoning survives.

## BM25 written out rather than imported

The baseline every later result is measured against should be understood rather than
trusted. Practically: `rank_bm25` and most pip BM25 packages implement the Robertson
IDF, `ln((N - df + 0.5) / (df + 0.5))`, which goes negative for any term appearing in
more than half the corpus. Lucene adds 1 inside the log so it cannot.

**Correction (experiment 10).** This note originally said the published BEIR baselines
come from Elasticsearch. They do not. The BEIR paper states: "We use Anserini with the
default Lucene parameters (k=0.9 and b=0.4)", and adds that they "also tested
Elasticsearch BM25 and Anserini + RM3 expansion, but found Anserini BM25 to perform the
best." Anserini is Lucene-based, so the argument for the Lucene IDF variant is unchanged
and if anything stronger — but the attribution was wrong, and it was wrong in the
document whose job is to record why the choice was made.

## `k1=0.9, b=0.4` rather than Lucene's defaults

Lucene ships `k1=1.2, b=0.75`. The BEIR paper used `0.9/0.4` for its baselines. Since
the point of the baseline is comparability with published numbers, BEIR's settings
win. Both are recorded with every run, because scores across the two are not
comparable and it is easy to forget which produced what.

## Exponential gain in nDCG

`trec_eval`'s `ndcg_cut` uses `2^rel - 1`. For binary qrels it makes no difference at
all — `2¹ - 1 = 1` — which is why SciFact can be scored either way and why it is a
safe dataset to build the harness on. For graded qrels it matters, so both are
implemented and the choice is recorded rather than defaulted silently.

## Unjudged documents count as non-relevant

This is what `trec_eval` does and what the BEIR leaderboard assumes. It is also a real
limitation: on sparsely judged datasets a genuinely good result that nobody judged is
scored as a miss. Worth stating whenever a number is quoted, rather than treating
nDCG@10 as ground truth.

## Zero-scoring documents are dropped from the ranking

BM25 returns only documents sharing at least one query term. Padding the ranking out
to `top_k` with zero-scoring documents would add nothing but tie-breaking noise, and
recall@100 is unaffected since those documents contribute nothing either way.

## Ties broken by document id

Retrieval scores tie more often than people expect, especially with short queries.
Breaking ties by id makes a run deterministic regardless of dict ordering upstream.
`pytrec_eval` breaks them differently, so cross-tool comparisons can differ in the
fourth decimal — not worth chasing, worth knowing about.

## Stemming optional, not assumed

Stemming moves BEIR nDCG@10 by a point or two, which is the same order as the
differences an ablation is trying to detect. Leaving it implicit would make later
comparisons meaningless. It is a runtime setting, recorded with every result, and
`snowballstemmer` is an optional extra so the core package needs only numpy.

## Paired randomisation test rather than bootstrap or t-test

Comparing two runs on the same queries is a paired problem, and the pairing carries most
of the evidence: two systems can differ by a tenth of a point on the mean while
disagreeing on a third of the queries, or move sixty queries and still be a coin flip.

A t-test is the conventional choice and the least comfortable one here. Per-query nDCG
is bounded in [0, 1] and piles up on exactly 0 and exactly 1 — on SciFact a majority of
queries score identically under two configurations — so the normality it assumes is
visibly wrong. Bootstrap is defensible and would also give confidence intervals on the
effect, but there is no exact answer to check a bootstrap against.

The randomisation test has the property that mattered most for this repo: for small
query counts every sign assignment can be enumerated, so the sampled implementation can
be tested against an exhaustively computed ground truth rather than against itself.
That is the same argument as `naive_bm25` in the BM25 tests, and `tests/test_significance.py`
uses it. Smucker, Allan and Carterette (2007) compared the candidates on IR data and
treat randomisation as the reference; that it needs only numpy settled it.

Two-sided by default. "Is A better than B" presumes the direction before looking, which
is the wrong question for an ablation.

## Holm-Bonferroni across a family of comparisons

A 2x2 ablation yields four comparisons off one set of 300 queries, and at a 0.05
threshold roughly one in twenty independent tests clears it by chance. Holm-Bonferroni
controls the chance of any false positive across the family, is uniformly less
conservative than plain Bonferroni, and assumes nothing about dependence between tests —
which matters, because comparisons sharing a configuration are certainly not independent.

The practical consequence is worth stating plainly: it changes conclusions. In experiment
2 the stemming effect on nDCG@10 has a raw p of 0.055 and 0.028, and adjusts to 0.166 and
0.110. Reporting the raw values would have made stemming look like an established win on
ranking. The family size is therefore recorded with every comparison, because a p-value
adjusted across four tests is not the same number as one adjusted across two.

## Per-query scores are committed, next to the summary rather than inside it

A paired test cannot be re-run from mean scores, so per-query values have to survive in
the repository or every significance claim becomes unreproducible from a clean clone.
They also would have buried the summary files, which exist to be read: 300 queries times
six metrics against a dozen lines of settings.

So `results/<run>.json` keeps the summary and gains a `per_query_file` pointer, and
`results/per-query/<run>.json` holds the scores. Both are committed. This is distinct
from `results/runs/`, which is gitignored and reserved for full rankings — those are
large, and nothing downstream needs them to reproduce a number.


## Dense retrieval is an optional extra, and torch pins numpy

`sentence-transformers` pulls in torch, which is roughly two orders of magnitude larger
than everything else here combined. The core package stays on numpy and tqdm, and dense
and hybrid retrieval live behind `pip install -e '.[dense]'`, the same shape as stemming.

There is a constraint worth writing down because it cost time to find. On this machine
pip offers torch only up to 2.2.2 — the conda environment's Python is x86_64 running
under Rosetta, and torch dropped x86 macOS wheels after that release. torch 2.2.2
predates numpy 2 and fails to initialise against it, which surfaces as
`Failed to initialize NumPy: _ARRAY_API not found` followed by transformers deciding
torch is absent entirely and raising a `NameError` from an unrelated module. The fix is
`numpy<2` with `transformers==4.40.2` and `sentence-transformers==2.7.0`.

Downgrading numpy under an evaluation harness is exactly the kind of change that can
move results silently, so it was verified rather than assumed: the SciFact BM25 baseline
re-run under numpy 1.26.4 reproduces 0.6802137133984872 bit-for-bit, and all tests pass.
Every results file now records its numpy version alongside its Python version, because
this episode demonstrates that it is a variable and not a constant.

## Query-type analysis uses one pre-specified continuous predictor

The obvious way to test "retrieval method is query-dependent" is to label queries by
type and compare group means. It is also the easiest way to find an effect that is not
there: labelling schemes are cheap to invent, and with enough of them one will separate
the systems by chance.

So the claim was reduced to a single hypothesis, fixed in the script before it was run:
the per-query advantage of BM25 over dense correlates positively with `max_idf`, the
rarity of the query's rarest term. One predictor, one test, one direction predicted in
advance. `max_idf` is computable from the query and the index alone, before any
retrieval happens, which is what stops the correlation being circular — a predictor
derived from how a system performed would guarantee its own result.

Other predictors (`mean_idf`, out-of-vocabulary rate, query length) are computed and
reported, but labelled secondary and Holm-adjusted among themselves. The primary is not
adjusted against them: penalising a hypothesis fixed in advance for tests invented
afterwards gets the logic backwards. The distinction is the only thing that keeps either
number meaningful, and query length on NFCorpus is a live example — it correlates more
strongly than the primary predictor and was not predicted, so it is recorded as a lead
for a pre-registered test on a third dataset rather than as a finding.


## Documentation figures are tested against the results files

The project's first rule is that every number in the README comes from a run. That was
enforced by remembering to do it, which is the kind of invariant that rots quietly: a
re-run shifts a figure, the results file updates, the prose does not.

The first version of the check was weaker than it looked. It asked whether each headline
number appeared *somewhere* in the README, and it passed when a value was deliberately
corrupted — because the same figure appeared in another table and the substring still
matched. A test that cannot fail is worse than no test, because it converts an unchecked
invariant into one everybody believes is checked.

So the question is inverted. `tests/test_documentation.py` extracts every three- and
four-decimal figure from the README and both documents in `docs/`, and requires each to
be traceable to a value in `results/` — as a metric, a recorded delta, or a percentage
of one. A fabricated number has nowhere to come from and fails immediately. Figures that
are legitimately not measurements — BEIR's published baselines, thresholds quoted in
prose — are listed individually with a reason, because a bare allowlist is how this
check would quietly be defanged.

Running it found two real violations rather than hypothetical ones. The TREC-COVID
query-formulation table and the recall-ceiling table were both quoted in the README but
produced by one-off commands that were never committed. Both are now scripts
(`diagnose_query_fields.py`, and `oracle_recall_at_k` recorded with every run), and the
numbers reproduce exactly. The rule now holds because it is checked, not because it was
followed carefully.


## Documents are concatenated, and BEIR does not do that

`BM25Retriever` indexes `title + " " + text` as one bag of words. The docstring used to
claim this matched BEIR's convention. It does not, and nobody checked until experiment 10
went looking for the cause of the TREC-COVID reproduction failure.

BEIR's paper is explicit: "We index the title (if available) and passage as separate
fields for documents." Under Lucene that means each field carries its own document
lengths, its own average length and its own document frequencies, and a query scores
against each field independently before the scores are summed. Concatenation collapses
all of that into one distribution.

The difference is largest where documents are lopsided. A title-only document — 24.6% of
TREC-COVID — is a very short *document* under concatenation, so length normalisation
inflates whatever it matches. Split into fields it is an ordinary-length title plus an
empty body that contributes nothing.

`MultiFieldBM25Retriever` implements the BEIR arrangement and experiment 10 measures what
it is worth. Concatenation remains the default rather than being silently swapped,
because every result already in this repository was produced with it and changing the
default would invalidate them all at once; the migration is a decision to take
deliberately, with the numbers in hand.
