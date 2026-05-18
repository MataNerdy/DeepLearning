import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import catboost
from preprocessing import clean_data, split_features_target, CAT_COLS
from metrics import evaluate_roc_auc, print_scores


def train_catboost(train_path, model_path):
    data = pd.read_csv(train_path)
    data = clean_data(data)

    X, y = split_features_target(data)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = catboost.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        l2_leaf_reg=0.894736842105263,
        cat_features=CAT_COLS,
        eval_metric="AUC",
        random_state=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    train_auc, valid_auc = evaluate_roc_auc(
        model, X_train, y_train, X_valid, y_valid
    )

    print_scores("CatBoost", train_auc, valid_auc)

    joblib.dump(model, model_path)


if __name__ == "__main__":
    train_catboost(
        train_path="data/train.csv",
        model_path="outputs/models/catboost.joblib"
    )