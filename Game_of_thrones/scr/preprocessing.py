from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def fill_missing_values(df):
    df = df.copy()

    categorical_cols = ["house", "culture", "title"]

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    df["age"] = df["age"].fillna(df["age"].median())

    return df

def create_features(df):
    df = df.copy()

    df["isPopular"] = (df["popularity"] > 0.5).astype(int)

    df["boolDeadRelations"] = (
        df["numDeadRelations"] > 0
    ).astype(int)

    return df

def build_preprocessor(categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    return preprocessor