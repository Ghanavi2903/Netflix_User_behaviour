"""Data loading, cleaning, encoding and train/test splitting."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churned"

# Derived monthly fee mapping (no explicit fee column in the raw CSV)
SUBSCRIPTION_FEE = {"Basic": 9.99, "Standard": 15.49, "Premium": 22.99}

NUMERICAL_COLS = [
    "age", "account_age_months", "session_count",
    "avg_watch_time_minutes_per_week", "watch_sessions_per_week",
    "completion_rate", "avg_rating_given", "app_rating",
    "recommendation_click_rate", "days_since_last_login",
    "monthly_fee",
]

CATEGORICAL_COLS = [
    "gender", "region", "subscription_type", "payment_method",
    "primary_device", "favorite_genre", "time_of_day",
    "recommendation_source",
]

DROP_COLS = ["user_id"]


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and add derived columns."""
    df = pd.read_csv(path)
    df = add_derived_columns(df)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create `monthly_fee` from subscription_type."""
    if "monthly_fee" not in df.columns and "subscription_type" in df.columns:
        df["monthly_fee"] = df["subscription_type"].map(SUBSCRIPTION_FEE)
    return df


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Column / dtype / non-null summary."""
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    return info


def data_quality(df: pd.DataFrame) -> dict:
    """Return missing values and duplicate stats."""
    missing = df.isna().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    dup_count = int(df.duplicated().sum())
    dup_id_count = int(df.duplicated(subset=["user_id"]).sum()) if "user_id" in df else 0
    return {
        "missing": missing[missing["missing_count"] > 0],
        "duplicate_rows": dup_count,
        "duplicate_user_ids": dup_id_count,
    }


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, mode, std for numerical columns."""
    num = df.select_dtypes(include=np.number)
    stats = pd.DataFrame({
        "count": num.count(),
        "mean": num.mean().round(2),
        "median": num.median().round(2),
        "mode": num.apply(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        "std": num.std().round(2),
        "min": num.min(),
        "max": num.max(),
        "q1": num.quantile(0.25),
        "q3": num.quantile(0.75),
    })
    return stats.round(2)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact dups, fill numeric medians, cat modes."""
    df = df.copy()
    df = df.drop_duplicates()
    if "user_id" in df:
        df = df.drop_duplicates(subset=["user_id"], keep="first")
    for c in df.select_dtypes(include=np.number).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include="object").columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df.reset_index(drop=True)


def encode_and_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Build ColumnTransformer, encode, split. Returns X_train, X_test, y_train, y_test, preprocessor, feature_names."""
    df = clean_data(df)
    # Keep only the columns we want
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    feature_cols = num_cols + cat_cols

    X = df[feature_cols].copy()
    y = df[TARGET].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    X_enc = preprocessor.fit_transform(X)
    # Feature names for display
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor, feature_names