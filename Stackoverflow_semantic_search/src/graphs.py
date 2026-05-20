import pandas as pd
import matplotlib.pyplot as plt

def show_retrieval_examples(
    queries,
    candidates,
    true_indices,
    predicted_rankings,
    n_good=5,
    n_bad=5,
    top_k=5,
):
    """
    queries: list[str] — тексты query-вопросов
    candidates: list[str] — база candidate-вопросов
    true_indices: list[int] — индекс правильного candidate для каждого query
    predicted_rankings: list[list[int]] — отсортированные индексы candidates для каждого query
    """

    good_examples = []
    bad_examples = []

    for i, ranking in enumerate(predicted_rankings):
        true_idx = true_indices[i]

        example = {
            "query": queries[i],
            "true_answer": candidates[true_idx],
            "top_predictions": [candidates[j] for j in ranking[:top_k]],
            "true_rank": ranking.index(true_idx) + 1 if true_idx in ranking else None,
        }

        if ranking[0] == true_idx:
            good_examples.append(example)
        else:
            bad_examples.append(example)

    return good_examples[:n_good], bad_examples[:n_bad]

metrics_df = pd.DataFrame([
    {
        "experiment": "pretrained_simple_tokenizer",
        "Hits@1": 0.285,
        "Hits@5": 0.393,
        "Hits@10": 0.449,
        "DCG@10": 0.360,
    },
    {
        "experiment": "pretrained_custom_tokenizer",
        "Hits@1": 0.407,
        "Hits@5": 0.575,
        "Hits@10": 0.641,
        "DCG@10": 0.516,
    },
    {
        "experiment": "trained_w2v_normalized",
        "Hits@1": 0.370,
        "Hits@5": 0.551,
        "Hits@10": 0.624,
        "DCG@10": 0.487,
    },
])

plot_df = metrics_df.set_index("experiment")

ax = plot_df.plot(kind="bar", figsize=(12, 6))
plt.title("Semantic Retrieval Quality Comparison")
plt.ylabel("Score")
plt.xlabel("Experiment")
plt.xticks(rotation=0, ha="center")
plt.ylim(0, 0.75)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("semantic_retrieval_comparison.png")

plt.figure(figsize=(12, 6))
plot_df.plot(kind="bar", ax=plt.gca())

plt.title("Semantic Retrieval Quality Comparison")
plt.ylabel("Score")
plt.xlabel("Experiment")
plt.xticks(rotation=0, ha="center")
plt.ylim(0, 0.75)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig("metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

def print_examples(examples, title):
    print("=" * 100)
    print(title)
    print("=" * 100)

    for idx, ex in enumerate(examples, 1):
        print(f"\nExample {idx}")
        print("-" * 100)
        print("QUERY:")
        print(ex["query"])

        print("\nTRUE ANSWER:")
        print(ex["true_answer"])

        print(f"\nTRUE RANK: {ex['true_rank']}")

        print("\nTOP PREDICTIONS:")
        for rank, pred in enumerate(ex["top_predictions"], 1):
            marker = "+" if pred == ex["true_answer"] else "  "
            print(f"{marker} {rank}. {pred}")
