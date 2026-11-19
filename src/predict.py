"""Prediction service for the Flask website.

This module turns one passenger's raw details into the same 10-feature row used
during training, loads the trained Random Forest (training it once if the model
file is missing) and returns a survival probability.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from .pipeline import (
    MODEL_PATH,
    OUTPUT_DIR,
    build_model,
    prepare_data,
)

# Used only if the saved model cannot report its own feature names.
DEFAULT_FEATURE_ORDER = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]


@lru_cache(maxsize=1)
def get_model():
    """Load the saved forest, or train and save it on first use."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    _, X, y = prepare_data()
    model = build_model().fit(X, y)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def model_feature_names() -> list[str]:
    """Return the exact feature order used by the trained model."""
    model = get_model()
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else DEFAULT_FEATURE_ORDER


def build_feature_row(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
) -> dict[str, float]:
    """Apply the project's feature engineering to one passenger."""
    family_size = sibsp + parch + 1
    return {
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "FamilySize": family_size,
        "IsAlone": 1 if family_size == 1 else 0,
        "Sex_male": 1 if sex == "male" else 0,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0,
    }


def predict_passenger(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
) -> dict:
    """Return the prediction, probability and an explanation for one passenger."""
    model = get_model()
    row = build_feature_row(pclass, sex, age, sibsp, parch, fare, embarked)

    feature_names = model_feature_names()
    X_one = pd.DataFrame([row], columns=feature_names)

    probabilities = model.predict_proba(X_one)[0]
    class_index = list(model.classes_).index(1)
    survival_probability = float(probabilities[class_index])
    survived = survival_probability >= 0.5

    return {
        "survived": survived,
        "prediction": "Survived" if survived else "Did not survive",
        "probability": round(survival_probability * 100, 1),
        "confidence": round(
            (survival_probability if survived else 1 - survival_probability) * 100,
            1,
        ),
        "family_size": row["FamilySize"],
        "is_alone": bool(row["IsAlone"]),
        "features": row,
        "sex": sex,
        "embarked": embarked,
    }
