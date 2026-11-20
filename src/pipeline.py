"""Reusable data + model helpers for the Titanic survival prediction project.

Keeping the pipeline in one small module means the terminal script and the
notebook use the same logic and cannot silently drift apart.

Pipeline (in order):
    1. Load the 891-row Kaggle training file.
    2. Clean missing values (drop Cabin, mode for Embarked, median for Age).
    3. Engineer FamilySize and IsAlone.
    4. One-hot encode Sex and Embarked (drop_first=True).
    5. Select model features and target.
    6. Split 80/20 and train a 100-tree Random Forest.
"""

from __future__ import annotations

import urllib.request
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Project paths and reproducibility settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODEL_PATH = OUTPUT_DIR / "titanic_random_forest.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.txt"

# The project reproduces Titanic2.ipynb exactly: one fixed seed (42) for both
# the train/test split and the forest.  It yields the notebook's 179-row test
# set, 82.68% accuracy and the [[91, 14], [17, 57]] confusion matrix.
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Public mirror of the standard Kaggle train.csv (891 rows x 12 columns).
# Used only when data/train.csv is missing.
MIRROR_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

REQUIRED_COLUMNS = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]

# Columns that are identifiers or free text, so they are not model features.
NON_PREDICTIVE_COLUMNS = ["Name", "Ticket", "PassengerId"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def download_dataset(destination: Path = DATA_PATH) -> Path:
    """Download the standard Kaggle Titanic train.csv to ``destination``.

    Raises a clear, actionable error if the download fails so the user can
    fall back to a manual Kaggle download.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"data/train.csv not found. Downloading a copy from:\n  {MIRROR_URL}")
    try:
        urllib.request.urlretrieve(MIRROR_URL, destination)
    except Exception as exc:  # noqa: BLE001 - show the real reason to the user
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the dataset automatically.\n"
            "Please download train.csv from https://www.kaggle.com/c/titanic/data\n"
            f"and save it as: {destination}"
        ) from exc
    print(f"Saved dataset to: {destination}")
    return destination


def load_titanic_data(
    csv_path: str | Path = DATA_PATH, allow_download: bool = True
) -> pd.DataFrame:
    """Load train.csv, optionally downloading the public mirror if missing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}.\n"
                "Place the Kaggle train.csv there or run without --no-download."
            )
        download_dataset(csv_path)

    df = pd.read_csv(csv_path)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing_columns}. "
            "Use the standard Kaggle Titanic train.csv."
        )

    if df.shape != (891, 12):
        warnings.warn(
            f"Expected the standard 891 x 12 Kaggle file, but got "
            f"{df.shape[0]} x {df.shape[1]}. Metrics will differ.",
            stacklevel=2,
        )
    return df


# ---------------------------------------------------------------------------
# Cleaning and feature engineering
# ---------------------------------------------------------------------------

def missing_value_counts(df: pd.DataFrame) -> pd.Series:
    """Return missing-value counts for every column that has any missing."""
    counts = df.isnull().sum()
    return counts[counts > 0].sort_values(ascending=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values exactly as described in the presentation.

    - Cabin: ~77% missing -> drop the column instead of inventing values.
    - Embarked: 2 missing -> fill with the most frequent port (mode = S).
    - Age: 177 missing -> fill with the median (robust to outliers).
    """
    df = df.copy()

    df = df.drop(columns=["Cabin"])

    embark_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embark_mode)

    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize and IsAlone from the existing SibSp/Parch columns."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categorical columns and return (X, y).

    ``drop_first=True`` keeps one column per category pair (e.g. Sex_male
    instead of both Sex_female and Sex_male), which avoids redundant columns.
    """
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    drop_columns = ["Survived", *NON_PREDICTIVE_COLUMNS]
    X = df.drop(columns=drop_columns)
    y = df["Survived"]
    return X, y


def prepare_data(
    csv_path: str | Path = DATA_PATH, allow_download: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run loading -> cleaning -> feature engineering -> encoding."""
    raw_df = load_titanic_data(csv_path, allow_download=allow_download)
    cleaned_df = clean_data(raw_df)
    featured_df = add_features(cleaned_df)
    X, y = encode_features(featured_df)
    return featured_df, X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into 80% train / 20% test using the project seed (42)."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)


def build_model(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """Create the notebook's 100-tree Random Forest."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=random_state,
    )
