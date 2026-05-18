import pandas as pd
import joblib

from preprocessing import clean_data


def make_submission(test_path, model_path, output_path):
    test = pd.read_csv(test_path)
    test = clean_data(test)

    model = joblib.load(model_path)

    predictions = model.predict_proba(test)[:, 1]

    submission = pd.DataFrame({
        "Id": test.index,
        "Churn": predictions
    })

    submission.to_csv(output_path, index=False)


if __name__ == "__main__":
    make_submission(
        test_path="data/test.csv",
        model_path="outputs/models/logreg.joblib",
        output_path="outputs/submissions/submission.csv"
    )