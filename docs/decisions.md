# Design decisions

Short notes on choices that are not obvious from the code, so the reasoning survives.

## BM25 written out rather than imported

The baseline every later result is measured against should be understood rather than
trusted. Practically: `rank_bm25` and most pip BM25 packages implement the Robertson
IDF, `ln((N - df + 0.5) / (df + 0.5))`, which goes negative for any term appearing in
more than half the corpus. Lucene adds 1 inside the log so it cannot. The published
BEIR baselines come from Elasticsearch, so matching them means matching Lucene.

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
