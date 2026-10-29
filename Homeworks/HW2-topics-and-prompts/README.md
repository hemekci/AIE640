# HW2: Topics and Prompts

**Covers:** Weeks 5–6 (book chapters 5–6) · **Due:** Week 7, 12 Nov 2026, 23:59 (Turkey time) · **Weight:** 5% · **Expected effort:** 6–8 hours

Individual work. Runs on Google Colab with a free T4 GPU. Course policy: AI tools may be used for learning but not to complete the homework.

## The question

A research group asks you two things. First, what is the NLP literature about, and how far can a topic map be trusted? Second, can they label movie reviews by prompting a small open model instead of training a classifier like the ones you built in HW1?

## Part A: A topic map of NLP research (Chapter 5) — 40 points

Data: the first 10,000 rows of [`MaartenGr/arxiv_nlp`](https://huggingface.co/datasets/MaartenGr/arxiv_nlp) (`train` split, column `Abstracts`).

1. Embed the abstracts with `thenlper/gte-small` (sentence-transformers).
2. Reduce with UMAP (`n_components=5`, `min_dist=0.0`, `metric="cosine"`, `random_state=42`) and cluster with HDBSCAN (`min_cluster_size=50`, `metric="euclidean"`, `cluster_selection_method="eom"`). Report the number of clusters and the share of documents marked as outliers (label −1).
3. **Stability.** Repeat step 2 with `random_state` 0, 1 and 2 (same embeddings). Report the adjusted Rand index (`sklearn.metrics.adjusted_rand_score`, all labels including −1) for the three pairs of runs, and the number of clusters in each run.
4. Fit BERTopic with the same embeddings, UMAP and HDBSCAN settings (`random_state=42`). Show the 10 largest topics with their size and top 5 words. Then update the representation with `KeyBERTInspired` and compare the words of 3 topics before and after.
5. In 4–6 sentences: what are the main research areas? Which topics look like artefacts (outliers, generic words, duplicates)? Given step 3, how much would you trust "there are N topics"?

## Part B: Prompting as a classifier (Chapter 6) — 45 points

Model: `Qwen/Qwen2.5-1.5B-Instruct` in half precision with **greedy decoding** (`do_sample=False`), so your results do not depend on a seed. Use the chat template.

Data: 200 reviews from the `test` split of [`rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes): `test.shuffle(seed=42).select(range(200))`. Any examples you put into a prompt must come from `train`.

Compare four prompting strategies on these 200 reviews:

1. **Zero-shot:** an instruction and the review; the answer is one word.
2. **Few-shot:** the same instruction with 4 labelled examples from `train` (2 positive, 2 negative).
3. **Chain-of-thought:** ask for brief reasoning, then a final line `Answer: positive` or `Answer: negative` (`max_new_tokens=200`).
4. **Structured output:** ask for JSON `{"sentiment": "positive" | "negative", "confidence": 0–1}` and parse it.

Use one parsing function for all four. An output from which no label can be extracted counts as wrong. For each strategy report accuracy, macro F1, **parse rate** (share of outputs with a valid label) and average seconds per review. Then evaluate your HW1 approach *embeddings + logistic regression* on the same 200 reviews, and show 3 reviews where chain-of-thought changed the zero-shot answer (from right to wrong or wrong to right).

## Part C: Recommendation — 15 points

Half a page in the notebook. Would you label the reviews by prompting, and with which strategy, or use the HW1 classifier? Use your numbers: quality, time and output length, reliability (parse rate), and the need for labelled data. Briefly: what can the topic map from Part A do for this research group, and what can it not?

## Submission (one zip, fixed names)

Upload `hw2.zip` through the [submission portal](https://yunus.hacettepe.edu.tr/~hakanemekci/submit/) (course AIE640, assignment HW2). It must contain:

| File | Content |
|---|---|
| `hw2.ipynb` | Your notebook, **run top to bottom with all outputs visible** |
| `results.json` | Your main numbers in the format below |

```json
{
  "clustering": {"n_clusters": 0, "outlier_fraction": 0.0, "ari_seed_pairs": [0.0, 0.0, 0.0]},
  "bertopic": {"n_topics": 0, "largest_topic_size": 0},
  "prompting": {
    "zero_shot":        {"accuracy": 0.0, "macro_f1": 0.0, "parse_rate": 0.0, "seconds_per_review": 0.0},
    "few_shot":         {"accuracy": 0.0, "macro_f1": 0.0, "parse_rate": 0.0, "seconds_per_review": 0.0},
    "chain_of_thought": {"accuracy": 0.0, "macro_f1": 0.0, "parse_rate": 0.0, "seconds_per_review": 0.0},
    "json_output":      {"accuracy": 0.0, "macro_f1": 0.0, "parse_rate": 0.0, "seconds_per_review": 0.0},
    "hw1_embeddings_logreg": {"accuracy": 0.0, "macro_f1": 0.0}
  },
  "recommended": "zero_shot | few_shot | chain_of_thought | json_output | hw1_embeddings_logreg"
}
```

## Grading (100 points)

Points per part as above. Within each part: correctness 60%, code quality (readable, settings as specified, no hard-coded results) 25%, analysis and insights 15%. The numbers in `results.json` must match the notebook outputs. Numbers are checked with tolerances that allow for differences between GPUs and library versions; timings are not compared with a reference. Late policy: 10% per day, up to 3 days; not accepted after that.
