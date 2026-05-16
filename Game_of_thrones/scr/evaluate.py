from sklearn.metrics import accuracy_score


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)

    return {
        "accuracy": acc
    }