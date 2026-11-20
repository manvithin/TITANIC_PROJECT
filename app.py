"""Flask website for the Titanic survival prediction project.

Run from the project root:

    python app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache

# matplotlib (used only to regenerate EDA charts) needs a writable cache dir.
# On read-only serverless filesystems such as Vercel, point it at /tmp.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from sklearn.metrics import classification_report, confusion_matrix

from src.pipeline import (
    DATA_DIR,
    FIGURES_DIR,
    MODEL_PATH,
    build_model,
    load_titanic_data,
    missing_value_counts,
    prepare_data,
    split_data,
)
from src.predict import get_model, predict_passenger

app = Flask(__name__)

REPO_URL = "https://github.com/manvithin/TITANIC_PROJECT"

REQUIRED_FIGURES = [
    "survival_by_sex.png",
    "survival_by_pclass.png",
    "target_count.png",
    "age_distribution.png",
    "correlation_heatmap.png",
    "feature_importance.png",
]

COLUMN_DESCRIPTIONS = {
    "PassengerId": "Row identifier - not used as a model feature.",
    "Survived": "Target: 0 = died, 1 = survived.",
    "Pclass": "Passenger class: 1 = first, 2 = second, 3 = third.",
    "Name": "Passenger name - excluded as free text.",
    "Sex": "male / female - one-hot encoded to Sex_male.",
    "Age": "Age in years - 177 missing values filled with the median (28.0).",
    "SibSp": "Number of siblings or spouses aboard.",
    "Parch": "Number of parents or children aboard.",
    "Ticket": "Ticket number - excluded as an identifier.",
    "Fare": "Passenger fare.",
    "Cabin": "Cabin number - dropped because 687 values are missing.",
    "Embarked": "Port: C = Cherbourg, Q = Queenstown, S = Southampton.",
}

DEFAULT_FORM = {
    "pclass": "3",
    "sex": "male",
    "age": "28",
    "sibsp": "0",
    "parch": "0",
    "fare": "8.05",
    "embarked": "S",
}


# The published report evaluates on the 80/20 split generated with
# ``random_state=42`` and the 100-tree forest seeded with 847.  Reproducing
# that split here keeps the website's confusion matrix and classification
# report identical to the written report (82.68%, [[91, 14], [17, 57]]).
REPORT_SPLIT_SEED = 42


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    """Evaluate the saved model once and return everything the pages display.

    The evaluation model is trained on the training split only (no leakage),
    reproducing the published report exactly (82.68%, [[91, 14], [17, 57]]).
    Production predictions via ``get_model()`` still train on all data for
    the best real-world results.
    """
    _, X, y = prepare_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, random_state=REPORT_SPLIT_SEED
    )
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["died", "survived"], output_dict=True
    )
    accuracy = float(report["accuracy"])
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))

    return {
        "accuracy": accuracy * 100,
        "accuracy_short": f"{accuracy * 100:.2f}%",
        "correct": int(cm[0, 0] + cm[1, 1]),
        "test_total": int(cm.sum()),
        "train_total": len(X_train),
        "cm": cm.tolist(),
        "class_0": report["died"],
        "class_1": report["survived"],
        "baseline": baseline * 100,
        "improvement": (accuracy - baseline) * 100,
        "trees": model.n_estimators,
        "features": model.n_features_in_,
        "passengers": len(X),
    }


@lru_cache(maxsize=1)
def get_raw_dataframe():
    """Load the raw table once for the dataset page."""
    return load_titanic_data()


def ensure_artifacts() -> None:
    """Train the model / draw charts if a fresh clone is missing them.

    The chart step is imported lazily so that serverless deployments without
    matplotlib/seaborn installed (and with read-only disks) can still serve
    the pre-generated figures committed in ``outputs/figures``.
    """
    get_model()
    if all((FIGURES_DIR / name).exists() for name in REQUIRED_FIGURES):
        return

    from src.eda_plots import create_all_eda_plots, plot_feature_importance

    featured_df, X, _ = prepare_data()
    create_all_eda_plots(featured_df, FIGURES_DIR)
    model = get_model()
    plot_feature_importance(model.feature_importances_, list(X.columns), FIGURES_DIR)


def validate_form(form: dict[str, str]) -> tuple[dict, list[str]]:
    """Validate the HTML form and return cleaned keyword arguments + errors."""
    cleaned: dict = {}
    errors: list[str] = []

    try:
        pclass = int(form.get("pclass", ""))
        if pclass not in (1, 2, 3):
            errors.append("Passenger class must be 1, 2 or 3.")
        cleaned["pclass"] = pclass
    except ValueError:
        errors.append("Passenger class must be a number (1, 2 or 3).")

    sex = form.get("sex", "").strip().lower()
    if sex not in ("male", "female"):
        errors.append("Sex must be male or female.")
    cleaned["sex"] = sex

    try:
        age = float(form.get("age", ""))
        if not 0 <= age <= 120:
            errors.append("Age must be between 0 and 120.")
        cleaned["age"] = age
    except ValueError:
        errors.append("Age must be a number.")

    for field, label, upper in (
        ("sibsp", "Siblings/spouses", 15),
        ("parch", "Parents/children", 15),
    ):
        try:
            value = int(form.get(field, ""))
            if not 0 <= value <= upper:
                errors.append(f"{label} must be between 0 and {upper}.")
            cleaned[field] = value
        except ValueError:
            errors.append(f"{label} must be a whole number.")

    try:
        fare = float(form.get("fare", ""))
        if not 0 <= fare <= 600:
            errors.append("Fare must be between 0 and 600.")
        cleaned["fare"] = fare
    except ValueError:
        errors.append("Fare must be a number.")

    embarked = form.get("embarked", "").strip().upper()
    if embarked not in ("C", "Q", "S"):
        errors.append("Embarked must be C, Q or S.")
    cleaned["embarked"] = embarked

    return cleaned, errors


@app.context_processor
def inject_globals():
    return {
        "metrics": get_metrics(),
        "repo_url": REPO_URL,
        "year": 2026,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = DEFAULT_FORM.copy()
    errors: list[str] = []
    result = None

    if request.method == "POST":
        form.update({key: value for key, value in request.form.items()})
        cleaned, errors = validate_form(form)
        if not errors:
            result = predict_passenger(**cleaned)

    return render_template(
        "predict.html", form=form, errors=errors, result=result
    )


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


@app.route("/dataset")
def dataset():
    df = get_raw_dataframe()
    missing = missing_value_counts(df)
    preview = df.head(12).to_html(
        classes="data-table", index=False, border=0, na_rep=""
    )
    columns = [
        {"name": column, "description": COLUMN_DESCRIPTIONS.get(column, "")}
        for column in df.columns
    ]
    return render_template(
        "dataset.html",
        preview=preview,
        missing=missing.to_dict(),
        columns=columns,
    )


@app.route("/figures/<path:filename>")
def figure(filename: str):
    """Serve generated EDA charts without exposing the whole outputs folder."""
    if filename not in REQUIRED_FIGURES:
        abort(404)
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/download/dataset")
def download_dataset():
    return send_from_directory(DATA_DIR, "train.csv", as_attachment=True)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_exists": MODEL_PATH.exists(),
            "accuracy": get_metrics()["accuracy_short"],
        }
    )


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


# Guarantee that a fresh clone has a trained model and the chart files,
# whether the app is started with "python app.py" or "flask --app app run".
# Vercel deploys may fail on cold start if artifacts are missing, so we
# guard this call so a failed import does not bring down the whole app.
try:
    ensure_artifacts()
except Exception:  # pragma: no cover
    # Artifacts will be generated on first request if missing;
    # a failure here should not prevent the server from starting.
    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
