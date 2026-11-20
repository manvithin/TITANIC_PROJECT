"""End-to-end Titanic survival prediction pipeline.

Run from the project root:

    python src/train_model.py

The script loads data/train.csv (downloading a public mirror if it is
missing), cleans it, engineers features, creates EDA charts, trains a
100-tree Random Forest and prints the exact evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

try:  # Support both "python src/train_model.py" and "python -m src.train_model".
    from .eda_plots import create_all_eda_plots, plot_feature_importance
    from .pipeline import (
        DATA_PATH,
        FIGURES_DIR,
        METRICS_PATH,
        MODEL_PATH,
        N_ESTIMATORS,
        OUTPUT_DIR,
        RANDOM_STATE,
        TEST_SIZE,
        add_features,
        build_model,
        clean_data,
        encode_features,
        load_titanic_data,
        missing_value_counts,
        split_data,
    )
except ImportError:
    from eda_plots import create_all_eda_plots, plot_feature_importance
    from pipeline import (
        DATA_PATH,
        FIGURES_DIR,
        METRICS_PATH,
        MODEL_PATH,
        N_ESTIMATORS,
        OUTPUT_DIR,
        RANDOM_STATE,
        TEST_SIZE,
        add_features,
        build_model,
        clean_data,
        encode_features,
        load_titanic_data,
        missing_value_counts,
        split_data,
    )

EXPECTED_ACCURACY = 148 / 179
EXPECTED_CONFUSION_MATRIX = [[91, 14], [17, 57]]


def print_header(title: str) -> None:
    """Print a simple section header that is easy to follow in a terminal."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Titanic survival Random Forest."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to train.csv (default: data/train.csv).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download the dataset if it is missing.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip EDA chart generation (metrics are unaffected).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_header("TITANIC SURVIVAL PREDICTION - RANDOM FOREST")
    print(f"Random seed : {RANDOM_STATE} (fixed for the split and the forest)")
    print(f"Model       : RandomForestClassifier(n_estimators={N_ESTIMATORS})")
    print(f"Split       : {int((1 - TEST_SIZE) * 100)}% train / {int(TEST_SIZE * 100)}% test")

    # ---------------------------------------------------------------- load
    print_header("1. LOAD THE DATASET")
    data_path = args.data if args.data is not None else DATA_PATH
    raw_df = load_titanic_data(data_path, allow_download=not args.no_download)

    print(f"Rows x columns : {raw_df.shape[0]} x {raw_df.shape[1]}")
    print(f"Columns        : {', '.join(raw_df.columns)}")
    print("\nFirst 5 rows:")
    print(raw_df.head().to_string(index=False))

    # ---------------------------------------------------------- missing values
    print_header("2. CHECK MISSING VALUES (BEFORE CLEANING)")
    missing_before = missing_value_counts(raw_df)
    print(missing_before.to_string())
    print(
        "\nExpected: Age = 177, Cabin = 687, Embarked = 2. "
        "These match the standard Kaggle training file."
    )

    print("\nCleaning decisions:")
    print("  - Drop Cabin: 687 of 891 values (77%) are missing.")
    print("  - Fill Embarked with the mode (most frequent port).")
    print("  - Fill Age with the median (robust to outliers).")

    cleaned_df = clean_data(raw_df)
    missing_after = missing_value_counts(cleaned_df)
    print(f"\nMissing values after cleaning: {int(missing_after.sum())}")

    # ------------------------------------------------------- feature engineering
    print_header("3. FEATURE ENGINEERING AND ENCODING")
    featured_df = add_features(cleaned_df)
    print(f"FamilySize = SibSp + Parch + 1 -> range "
          f"{featured_df['FamilySize'].min()} to {featured_df['FamilySize'].max()}")
    print("IsAlone = 1 when FamilySize == 1, otherwise 0")
    print(f"  Alone passengers  : {int(featured_df['IsAlone'].sum())}")
    print(f"  With family       : {int((1 - featured_df['IsAlone']).sum())}")

    X, y = encode_features(featured_df)
    print(f"\nEncoded features ({X.shape[1]}): {', '.join(X.columns)}")
    print(f"Feature matrix : {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Target         : Survived (0 = died, 1 = survived)")

    # ------------------------------------------------------------------- EDA
    if not args.skip_plots:
        print_header("4. EXPLORATORY DATA ANALYSIS CHARTS")
        create_all_eda_plots(featured_df, FIGURES_DIR)
        for path in sorted(FIGURES_DIR.glob("*.png")):
            try:
                shown = path.relative_to(Path.cwd())
            except ValueError:
                shown = path
            print(f"  saved: {shown}")

    # ------------------------------------------------------------ split/model
    print_header("5. TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training rows : {len(X_train)}")
    print(f"Test rows     : {len(X_test)}")
    print(
        f"Test classes  : {int((y_test == 0).sum())} died, "
        f"{int((y_test == 1).sum())} survived"
    )

    print_header("6. TRAIN THE RANDOM FOREST")
    model = build_model()
    model.fit(X_train, y_train)
    print(f"Trained {N_ESTIMATORS} decision trees on {len(X_train)} passengers.")

    # ------------------------------------------------------------- evaluation
    print_header("7. EVALUATION ON THE 179-PASSENGER TEST SET")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    correct = int(round(accuracy * len(y_test)))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Died (0)", "Survived (1)"], digits=2
    )

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct  : {correct} of {len(y_test)} passengers")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"                 Predicted died   Predicted survived")
    print(f"Actual died      {cm[0, 0]:>12}   {cm[0, 1]:>18}")
    print(f"Actual survived  {cm[1, 0]:>12}   {cm[1, 1]:>18}")
    print("\nRaw confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    # A majority-class baseline helps explain why 82.68% is useful.
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"Majority-class baseline on this test set: {baseline:.2%}")

    # --------------------------------------------------------- save artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
        plot_feature_importance(model.feature_importances_, list(X.columns), FIGURES_DIR)

    joblib.dump(model, MODEL_PATH)
    with open(OUTPUT_DIR / "model_features.json", "w", encoding="utf-8") as fh:
        json.dump(list(X.columns), fh, indent=2)

    metrics_lines = [
        "TITANIC SURVIVAL PREDICTION - EVALUATION SUMMARY",
        "=" * 52,
        f"Dataset            : data/train.csv ({raw_df.shape[0]} rows, {raw_df.shape[1]} columns)",
        "Missing before     : Age 177, Cabin 687, Embarked 2",
        "Missing after      : 0",
        f"Model              : RandomForestClassifier(n_estimators={N_ESTIMATORS})",
        f"Random state       : {RANDOM_STATE}",
        f"Split              : 80/20 (test_size={TEST_SIZE})",
        f"Training samples   : {len(X_train)}",
        (
            f"Test samples       : {len(X_test)} "
            f"({int((y_test == 0).sum())} died, {int((y_test == 1).sum())} survived)"
        ),
        "",
        f"Accuracy           : {accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"Correct predictions: {correct} / {len(y_test)}",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        f"[[{cm[0, 0]:>3} {cm[0, 1]:>3}]",
        f" [{cm[1, 0]:>3} {cm[1, 1]:>3}]]",
        "",
        "Classification report:",
        report,
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines), encoding="utf-8")

    print_header("8. ARTIFACTS AND REPRODUCIBILITY CHECK")
    print(f"Model saved      : {MODEL_PATH}")
    print(f"Feature list     : {OUTPUT_DIR / 'model_features.json'}")
    print(f"Metrics summary  : {METRICS_PATH}")
    if not args.skip_plots:
        print(f"Charts           : {FIGURES_DIR}")

    accuracy_ok = abs(accuracy - EXPECTED_ACCURACY) < 1e-12
    matrix_ok = cm.tolist() == EXPECTED_CONFUSION_MATRIX
    if accuracy_ok and matrix_ok:
        print("\nReproducibility  : PASSED")
        print("  Accuracy matches 82.68% and the confusion matrix matches [[91, 14], [17, 57]].")
    else:
        print("\nReproducibility  : WARNING - results differ from the presentation.")
        print(f"  Expected accuracy {EXPECTED_ACCURACY:.4f}, got {accuracy:.4f}")
        print(f"  Expected matrix   {EXPECTED_CONFUSION_MATRIX}, got {cm.tolist()}")
        print("  Check the dataset file and scikit-learn version (see README).")

    print_header("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
