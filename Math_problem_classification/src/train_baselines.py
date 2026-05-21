from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import DataConfig, load_math_problem_data, make_train_test_split


def train_and_evaluate_baselines(output_dir: str | Path = "reports/baselines") -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DataConfig()
    df = load_math_problem_data(config)
    train_df, test_df = make_train_test_split(df, config)

    x_train = train_df[config.text_column].astype(str)
    y_train = train_df[config.label_column].astype(str)
    x_test = test_df[config.text_column].astype(str)
    y_test = test_df[config.label_column].astype(str)

    models = {
        "TF-IDF + Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
        "TF-IDF + LinearSVC": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
    }

    rows = []
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)

        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "weighted_f1": f1_score(y_test, pred, average="weighted"),
            }
        )

        Path(output_dir, f"{name.lower().replace(' ', '_').replace('+', 'plus')}_report.txt").write_text(
            classification_report(y_test, pred, zero_division=0),
            encoding="utf-8",
        )

    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result.to_csv(output_dir / "baseline_metrics.csv", index=False)
    print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/baselines")
    args = parser.parse_args()
    train_and_evaluate_baselines(args.output_dir)


if __name__ == "__main__":
    main()
