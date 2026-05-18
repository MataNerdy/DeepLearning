import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from preprocessing import clean_data, split_features_target, make_logreg_pipeline
from metrics import evaluate_roc_auc, print_scores


def train_linear(train_path, model_path):
    data = pd.read_csv(train_path)
    data = clean_data(data)

    X, y = split_features_target(data)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = make_logreg_pipeline()

    param_grid = {
        "logisticregression__C": [100, 50, 35, 30, 25, 20, 10, 5, 1, 0.5, 0.1]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    train_auc, valid_auc = evaluate_roc_auc(
        best_model, X_train, y_train, X_valid, y_valid
    )

    print("Best params:", grid_search.best_params_)
    print_scores("Logistic Regression", train_auc, valid_auc)

    joblib.dump(best_model, model_path)


if __name__ == "__main__":
    train_linear(
        train_path="data/train.csv",
        model_path="outputs/models/logreg.joblib"
    )