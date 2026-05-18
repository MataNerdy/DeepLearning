from sklearn.metrics import roc_auc_score


def evaluate_roc_auc(model, X_train, y_train, X_valid, y_valid):
    train_pred = model.predict_proba(X_train)[:, 1]
    valid_pred = model.predict_proba(X_valid)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred)
    valid_auc = roc_auc_score(y_valid, valid_pred)

    return train_auc, valid_auc


def print_scores(model_name, train_auc, valid_auc):
    print(f"{model_name}")
    print(f"ROC-AUC train: {train_auc:.4f}")
    print(f"ROC-AUC valid: {valid_auc:.4f}")