import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

NUM_COLS = ["ClientPeriod", "MonthlySpending", "TotalSpent"]

CAT_COLS = [
    "Sex", "IsSeniorCitizen", "HasPartner", "HasChild",
    "HasPhoneService", "HasMultiplePhoneNumbers",
    "HasInternetService", "HasOnlineSecurityService",
    "HasOnlineBackup", "HasDeviceProtection",
    "HasTechSupportAccess", "HasOnlineTV",
    "HasMovieSubscription", "HasContractPhone",
    "IsBillingPaperless", "PaymentMethod"
]

TARGET_COL = "Churn"


def clean_data(df):
    df = df.copy()
    df["TotalSpent"] = pd.to_numeric(df["TotalSpent"], errors="coerce")
    df["TotalSpent"] = df["TotalSpent"].fillna(0)
    return df


def split_features_target(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def make_logreg_pipeline():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    return Pipeline([
        ("preprocessing", preprocessor),
        ("logisticregression", model),
    ])