from sklearn.model_selection import train_test_split

from data_loading import load_data
from preprocessing import fill_missing_values, create_features
from grid_search import run_grid_search


TARGET = "isAlive"


def main():

    train_df, test_df = load_data(
        "data/train.csv",
        "data/test.csv",
    )

    train_df = fill_missing_values(train_df)
    train_df = create_features(train_df)

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results_df = run_grid_search(
        X_train,
        y_train,
        X_val,
        y_val,
        cv=5,
    )

    print(results_df)

    results_df.to_csv(
        "reports/grid_search_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()