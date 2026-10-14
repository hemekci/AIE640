# HW1: From Tokens to Classifiers

**Covers:** Weeks 1–4 (book chapters 1–4) · **Out:** 13 Oct 2026 · **Due:** Week 5, 22 Oct 2026, 23:59 (Turkey time) · **Weight:** 5% · **Expected effort:** 6–8 hours

Individual work. Runs on Google Colab with a free T4 GPU. Course policy: AI tools may be used for learning but not to complete the homework.

## The question

You are asked to recommend a sentiment classifier for a product team that also plans a Turkish version of the product. Before choosing a model, you check two things: how much a Turkish text costs in tokens compared with English, and which of the classification approaches from Chapter 4 gives the best quality for its cost.

## Part A: What does Turkish cost? (Chapter 2) — 25 points

Use the first 500 sentence pairs of the `test` split of [`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100), config `en-tr` (the same sentences in English and Turkish).

1. Tokenize both sides with four tokenizers: `gpt2`, `bert-base-multilingual-cased`, `dbmdz/bert-base-turkish-cased`, and `Qwen/Qwen2.5-1.5B-Instruct`.
2. For each tokenizer and language, report **fertility** (tokens per whitespace word, averaged over sentences) and the ratio of total Turkish to total English tokens.
3. Show how each tokenizer splits three Turkish words with many suffixes (e.g. *evlerimizden*, *gelemeyeceklermiş*, *Afyonkarahisarlılaştıramadıklarımızdan*).
4. In 4–6 sentences: which tokenizer is the most expensive for Turkish, and why? What does this mean for context length and API cost?

## Part B: Inside the generator (Chapters 1 and 3) — 25 points

Use `Qwen/Qwen2.5-1.5B-Instruct` in half precision.

1. For the prompt `The capital of Turkey is`, print the top 10 next tokens with their probabilities.
2. Generate 5 continuations of a prompt of your choice with greedy decoding and with temperature 0.7 and 1.5 (`top_p=1`, fixed seed). Describe the difference in 3–4 sentences.
3. Time the generation of 200 new tokens with `use_cache=True` and `use_cache=False` (3 runs each, report the mean). Explain the difference using what you learned about the key/value cache.

## Part C: Three ways to classify (Chapter 4) — 35 points

Dataset: [`rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes); train on `train`, report on `test` (do not tune on `test`; use `validation` for any choice you make).

1. **Task-specific model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` used directly (map its labels to positive/negative and state how you handle "neutral").
2. **Embeddings + classifier:** `sentence-transformers/all-mpnet-base-v2` embeddings with a logistic regression trained on `train`.
3. **Zero-shot with embeddings:** cosine similarity between each review and label descriptions you write; try at least two phrasings of the descriptions and pick one on `validation`.

For each approach report accuracy and macro F1 on `test`, the confusion matrix, and the time to classify the whole test set.

## Part D: Recommendation — 15 points

Half a page in the notebook: which approach do you recommend to the product team, and why? Use your numbers (quality, speed, need for labelled data) and Part A (what would change for Turkish?). Show 5 misclassified test reviews from your recommended approach and explain what they have in common.

## Submission (one zip, fixed names)

Upload `hw1.zip` through the [submission portal](https://yunus.hacettepe.edu.tr/~hakanemekci/submit/) (course AIE640, assignment HW1). It must contain:

| File | Content |
|---|---|
| `hw1.ipynb` | Your notebook, **run top to bottom with all outputs visible** |
| `results.json` | Your main numbers in the format below |

```json
{
  "fertility": {"gpt2": {"en": 0.0, "tr": 0.0}, "bert-base-multilingual-cased": {"en": 0.0, "tr": 0.0},
                "dbmdz/bert-base-turkish-cased": {"en": 0.0, "tr": 0.0}, "Qwen/Qwen2.5-1.5B-Instruct": {"en": 0.0, "tr": 0.0}},
  "kv_cache_seconds": {"with_cache": 0.0, "without_cache": 0.0},
  "classification": {"task_specific": {"accuracy": 0.0, "macro_f1": 0.0},
                     "embeddings_logreg": {"accuracy": 0.0, "macro_f1": 0.0},
                     "zero_shot": {"accuracy": 0.0, "macro_f1": 0.0}},
  "recommended": "task_specific | embeddings_logreg | zero_shot"
}
```

## Grading (100 points)

Points per part as above. Within each part: correctness 60%, code quality (readable, fixed seeds, no hard-coded results) 25%, analysis and insights 15%. The numbers in `results.json` must match the notebook outputs. Late policy: 10% per day, up to 3 days; not accepted after that.
