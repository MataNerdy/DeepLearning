import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        ),
        "gaussian_process": GaussianProcessClassifier(random_state=RANDOM_STATE),
        "gaussian_nb": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "svc": SVC(random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }


def get_param_grids():
    return {
        "logreg": {
            "C": [0.1, 1, 3, 10],
            "solver": ["liblinear"],
        },

        "random_forest": {
            "n_estimators": [100, 200, 400],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },

        "adaboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.5, 1.0],
            "estimator__max_depth": [1, 2, 3],
        },

        "gaussian_process": {
            "max_iter_predict": [50, 100, 200],
        },

        "gaussian_nb": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },

        "knn": {
            "n_neighbors": [5, 7, 9, 11, 13, 15, 21],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },

        "svc": {
            "C": [0.1, 1, 3, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },

        "decision_tree": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5, 10],
            "criterion": ["gini", "entropy"],
        },
    }

def run_grid_search(X_train, y_train, X_val, y_val, cv=5):
    models = get_models()
    param_grids = get_param_grids()

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_val)
        holdout_accuracy = accuracy_score(y_val, y_pred)

        results.append({
            "model": name,
            "best_cv_accuracy": grid.best_score_,
            "holdout_accuracy": holdout_accuracy,
            "best_params": grid.best_params_,
        })

        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        print(f"Holdout accuracy: {holdout_accuracy:.4f}")
        print(f"Best params: {grid.best_params_}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["holdout_accuracy", "best_cv_accuracy"],
        ascending=False,
    )

    return results_df