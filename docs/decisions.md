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
