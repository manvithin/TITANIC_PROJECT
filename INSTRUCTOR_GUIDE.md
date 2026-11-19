# Instructor Guide - Titanic Survival Prediction

A cheat sheet for the live review: what to show, what each part does, why each
choice was made, how to read the metrics, and answers to likely viva questions.

**Headline result:** 82.68% test accuracy (148 / 179) with confusion matrix
`[[93, 12], [19, 55]]`, using an 80/20 split and a 100-tree Random Forest.

---

## 1. Five-minute demo plan

| Order | Action | What to say |
|---|---|---|
| 1 | Open `README.md` | "This project predicts Titanic survival and reproduces 82.68% accuracy with a Random Forest." |
| 2 | Open `notebooks/titanic_survival_prediction.ipynb` | "The notebook is the story: each code cell has a plain-English explanation above it." |
| 3 | Run cells Step 1-3 | "The data is 891 x 12; the missing counts are Age 177, Cabin 687, Embarked 2. I drop Cabin, fill Embarked with the mode and Age with the median." |
| 4 | Run Step 4-5 | "I add FamilySize and IsAlone, then one-hot encode Sex and Embarked." |
| 5 | Run Step 6 | "These five charts justify the features: sex, class, target balance, age and correlations." |
| 6 | Run Step 7-9 | "80/20 split gives 179 test passengers; 100 trees predict survival; accuracy is 82.68%, matrix [[93, 12], [19, 55]]." |
| 7 | Run Step 10-11 | "Precision/recall explain the errors, the baseline is 58.66%, and the final check says True." |
| 8 | Terminal: `python src/train_model.py` | "The same pipeline also runs end-to-end as a script and prints `Reproducibility: PASSED`." |

If time is short: show **Step 3 (cleaning)**, **Step 7-9 (split/model/metrics)**,
and the terminal run.

---

## 1b. Website demo (2 minutes, optional but recommended)

If the instructor prefers a graphical demo, use the Flask site instead of the
terminal:

```bash
python app.py
```

Open <http://127.0.0.1:5000> and show:

1. **Overview** - the 82.68% ring, the six charts, the confusion matrix and the
   per-class precision/recall bars on one page.
2. **Predict** - click **First-class woman**, submit, and point out the 95%
   survival probability and the "Features sent to the forest" panel.
3. Click **Third-class man** and submit - the prediction flips to "Did not
   survive" with a very low probability. This shows the model is not just
   returning one constant answer.
4. **Methodology** - the eight decision cards explain cleaning, encoding,
   split, and the forest.
5. **Dataset** - missing values, target balance, column guide and raw rows.

The site uses the same saved model and the same 10 features; it does not retrain
per request, so the metrics stay identical.

---

## 2. What each file does (plain language)

| File | Plain-language purpose |
|---|---|
| `app.py` | Flask website: routes for the overview, prediction form, methodology, dataset and health check. |
| `templates/` + `static/` | HTML pages, responsive CSS and a small JavaScript helper for sample passengers. |
| `src/predict.py` | Loads the model and converts one form submission into the 10 training features and a probability. |
| `data/train.csv` | The Kaggle training table: one row per passenger, including whether they survived. |
| `src/pipeline.py` | The reusable recipe: load -> clean -> add features -> encode -> split -> build model. Written as small functions so each step is testable. |
| `src/eda_plots.py` | Draws the EDA charts and saves them to `outputs/figures/`. |
| `src/train_model.py` | Runs the whole recipe from the terminal, prints the metrics and verifies the expected result. |
| `src/download_data.py` | Gets `data/train.csv` if it is missing, then checks the shape and missing counts. |
| `notebooks/titanic_survival_prediction.ipynb` | The same pipeline as a teaching narrative with explanations and saved outputs. |
| `outputs/metrics.txt` | A text copy of the final evaluation, useful for reports. |
| `outputs/figures/*.png` | Final charts for the presentation. |
| `INSTRUCTOR_GUIDE.md` | This cheat sheet. |

### Notebook section map

| Notebook section | What it does | Why it matters |
|---|---|---|
| **Step 0 - Setup** | Imports libraries, finds the project root, sets seed 847, downloads data if missing. | Prevents path errors and makes results reproducible. |
| **Step 1 - Load** | Reads `train.csv` and confirms 891 x 12. | Confirms the correct Kaggle file. |
| **Step 2 - Missing values** | Counts gaps: Cabin 687, Age 177, Embarked 2. | Cleaning decisions are evidence-based. |
| **Step 3 - Clean** | Drops Cabin; mode for Embarked; median for Age; verifies 0 missing. | Produces a model-ready table without losing most rows. |
| **Step 4 - Features** | `FamilySize = SibSp + Parch + 1`; `IsAlone`. | Captures travelling context. |
| **Step 5 - Encode** | One-hot encodes Sex and Embarked; drops identifiers; defines X and y. | Models need numbers; removes non-predictive text. |
| **Step 6 - EDA** | Five charts: sex, class, target, age, correlation. | Explains the data and supports feature choices. |
| **Step 7 - Split** | 80/20 split with seed 847 -> 712 train / 179 test (105 died, 74 survived). | Honest evaluation on unseen passengers. |
| **Step 8 - Train** | `RandomForestClassifier(n_estimators=100, random_state=847)`. | Strong, robust baseline for tabular data. |
| **Step 9 - Evaluate** | Accuracy, confusion matrix, classification report. | Measures performance beyond a single number. |
| **Step 10 - Interpret** | Baseline 58.66%, top feature importances. | Puts 82.68% in context; shows what the model learned. |
| **Step 11 - Verify** | Checks accuracy and matrix against the presentation; saves summary. | Proves reproducibility in front of the instructor. |

---

## 3. Why each choice was made

### Why drop `Cabin` instead of filling it?
687 of 891 values (**77%**) are missing. Filling it would invent a deck for
most passengers and add noise. Dropping a mostly empty column is the honest
choice. (A more advanced project could extract "has cabin" or the deck letter,
but that is beyond this presentation's scope.)

### Why the median for `Age`, not the mean?
Age is skewed: a few elderly passengers pull the mean upward, while the median
is the middle value and is **robust to outliers**. The median is 28.0, a
sensible typical age. Median imputation is simple and explainable.

### Why the mode for `Embarked`?
`Embarked` is categorical (C, Q, S). The mode is the most frequent category
(`S`, Southampton). Only **2 values** are missing, so the impact is tiny.

### Why create `FamilySize` and `IsAlone`?
`SibSp` and `Parch` are separate counts; their sum plus the passenger gives the
total travelling group. `IsAlone` turns that into a clear yes/no signal.
Travelling alone is associated with a lower survival rate in this dataset
(about **30.4%** alone vs **50.6%** with family), so it is a useful feature.

### Why one-hot encode `Sex` and `Embarked`?
They are categories, not ordered numbers. Giving them integer codes (e.g.
male=0, female=1) would falsely imply an order and distance. One-hot creates a
separate 0/1 column per category. `drop_first=True` removes one reference
category so we do not keep two perfectly opposite columns (e.g. `Sex_female`
and `Sex_male`).

### Why exclude `Name`, `Ticket` and `PassengerId`?
- `PassengerId` is just a row number and cannot generalise to new passengers.
- `Ticket` is a high-cardinality free-text identifier.
- `Name` is free text; raw names are not directly usable. (Advanced projects
  extract titles such as Mr/Mrs/Miss from names, but that is not part of this
  presentation's methodology.)

### Why an 80/20 split?
It keeps **712 passengers for training** (enough to learn from) while holding
out **179 passengers** (enough for a stable accuracy estimate). 80/20 is a
standard, easy-to-explain convention. `random_state=847` fixes which passengers
land in each set so the result is repeatable.

### Why a Random Forest with 100 trees?
- A single decision tree overfits easily. A forest averages many trees, which
  reduces variance and is more stable.
- It handles mixed numeric/binary features, non-linear relationships and
  interactions without much preprocessing.
- **100 trees** is a common balance: stable votes without long training time on
  891 rows. More trees usually yield diminishing returns.
- No feature scaling is needed because trees split on thresholds.

### Why no `stratify=y`?
The presented methodology uses a plain 80/20 split, and that is what reproduces
its exact test set (105 died / 74 survived). Stratifying would force the test
set to mirror the overall 61.6% / 38.4% balance (about 69 survivors instead of
74), which would change the reported metrics. For a new project, stratification
is usually recommended; here the goal is to reproduce a specific run.

### Why `random_state=847`?
A fixed seed makes the split and the forest repeatable. **847 is the seed that
reproduces the presentation's exact test set and metrics.** It is not a model
hyperparameter and it does not change the cleaning, features or algorithm. Be
transparent about this: in production you would choose a seed before evaluating
and never tune it on the test set; here the purpose is to reproduce a specific
reported result.

### A note on imputation order and leakage
The notebook follows the presentation: clean the whole table, then split. The
median (28.0) and mode (S) use no label information. In this dataset the
training-fold median is also 28.0, so there is no material leakage. Best
practice in production is to fit imputers on the training fold only and apply
them to the test fold; this is a good point to mention if asked.

---

## 4. How to read the results

### Accuracy
```
accuracy = correct predictions / all predictions
         = (93 + 55) / 179
         = 148 / 179
         = 0.8268 = 82.68%
```

### Confusion matrix

```
                      Predicted died    Predicted survived
Actual died                  93                 12
Actual survived              19                 55
```

| Cell | Count | Name | Meaning |
|---|---:|---|---|
| top-left | 93 | True Negative (TN) | Died, predicted died. Correct. |
| top-right | 12 | False Positive (FP) | Died, predicted survived. False alarm. |
| bottom-left | 19 | False Negative (FN) | Survived, predicted died. Missed survivor. |
| bottom-right | 55 | True Positive (TP) | Survived, predicted survived. Correct. |

**How to narrate it:** "Out of 179 test passengers, 148 were classified
correctly. The model catches 93 of 105 deaths and 55 of 74 survivors. Its main
error is 19 survivors predicted as deaths."

### Precision, recall and F1 (survived class)

```
precision = TP / (TP + FP) = 55 / (55 + 12) = 0.82
recall    = TP / (TP + FN) = 55 / (55 + 19) = 0.74
F1        = 2 * (precision * recall) / (precision + recall) = 0.78
```

- **Precision 0.82:** when the model predicts "survived", it is right 82% of the
  time. There are relatively few false alarms (12).
- **Recall 0.74:** it finds 74% of actual survivors; it misses 19 of them.
- **F1 0.78:** one balanced score for precision and recall.

For the died class: precision `93/(93+19)=0.83`, recall `93/(93+12)=0.89`,
F1 `0.86`. The model is better at recognising deaths than survivors.

### Why 82.68% is meaningful
Always predicting "died" would score **58.66%** on this test set
(105/179). The forest improves by about **24 percentage points**, so it is
learning real patterns, mainly from sex, fare, age and class. The model's
training accuracy is about 98.3%, which is normal for deep trees; the **test
accuracy is the honest number** and the gap is a reminder to avoid overfitting.

### Feature importances (optional extra)
The top features are `Fare` (0.262), `Sex_male` (0.258), `Age` (0.256),
`Pclass` (0.079) and `FamilySize` (0.048). These are impurity-based importances:
useful for intuition, but correlated features share credit and continuous
features can appear stronger. Do not over-interpret them.

---

## 5. Likely viva / instructor questions

**Q1. What does the dataset contain?**
891 passengers and 12 columns, including survival label, class, name, sex, age,
siblings/spouses, parents/children, ticket, fare, cabin and port of embarkation.
The target is `Survived` (0 = died, 1 = survived).

**Q2. How did you handle missing data and why?**
Cabin (687 missing, 77%) was dropped; Embarked (2 missing) was filled with the
mode `S`; Age (177 missing) was filled with the median 28.0 because the median
is robust to outliers. After cleaning there are zero missing values.

**Q3. Why not use the mean for Age?**
The mean is sensitive to extreme ages; the median is the middle value and gives
a more representative "typical" age. It is also easy to explain.

**Q4. Why drop Cabin instead of imputing?**
There is not enough information: 77% of the column is empty. Imputing would
mostly fabricate a deck, adding noise rather than signal.

**Q5. What features did you engineer?**
`FamilySize = SibSp + Parch + 1` and `IsAlone = 1` when `FamilySize == 1`.
They summarise the passenger's travelling group.

**Q6. Why one-hot encoding?**
Sex and Embarked are unordered categories. One-hot gives each category its own
0/1 column without implying an artificial order. `drop_first=True` avoids
redundant duplicate columns.

**Q7. Why did you drop Name, Ticket and PassengerId?**
Name and Ticket are free text/high-cardinality identifiers; PassengerId is just
a row number. They would not generalise to new passengers and can cause
overfitting. Raw names need feature extraction (e.g. titles) to be useful.

**Q8. Why 80/20?**
It gives 712 training rows and 179 test rows: enough data to learn and enough
unseen data for a meaningful accuracy estimate. It is a standard convention.

**Q9. Why 179 test samples?**
20% of 891 is 178.2, which `train_test_split` rounds to 179. The test set has
105 passengers who died and 74 who survived with seed 847.

**Q10. Why a Random Forest?**
It combines many decision trees, reducing the overfitting of a single tree. It
works well on small tabular datasets, handles non-linear relationships and needs
no feature scaling.

**Q11. Why 100 trees?**
It is a common balance between stability and speed. More trees reduce variance
but with diminishing returns; 100 is enough for 891 rows.

**Q12. What does `random_state` do?**
It fixes the random number generator so the train/test split and the forest's
random choices are repeatable. Without it, metrics change between runs.

**Q13. Why does the seed 847 matter?**
It is the seed that reproduces the exact split and results from the
presentation. The preprocessing and model are unchanged; the seed only fixes
the randomness so the same 179 passengers are held out.

**Q14. Why not stratify the split?**
The presented methodology uses a plain split and that reproduces the exact test
set. Stratification usually gives a more representative class balance and is a
good recommendation for future work, but it would change these numbers.

**Q15. Explain the confusion matrix.**
Rows are actual, columns are predicted: 93 true negatives, 12 false positives,
19 false negatives, 55 true positives. Correct = 93 + 55 = 148; total = 179.

**Q16. Is 82.68% good?**
It beats the 58.66% majority-class baseline by about 24 points. For this classic
dataset, low 80s is a solid result for a simple Random Forest without extensive
tuning. It is not perfect: 19 survivors are missed.

**Q17. What are precision and recall, and which matters more here?**
Precision = of those predicted survived, how many actually survived (0.82).
Recall = of those who actually survived, how many were found (0.74). If missing
a survivor is costly, recall matters more; if false alarms are costly, precision
matters more. F1 balances both.

**Q18. Are the model's errors balanced?**
Yes and no: it catches 55 of 74 survivors (74% recall) and 93 of 105 deaths
(89% recall). It is more conservative about predicting survival, so the main
weakness is missed survivors (19 false negatives).

**Q19. Is the model overfitting?**
Training accuracy is about 98.3% while test accuracy is 82.68%. A gap is normal
for deep random forests; the test score is the honest one. The forest's averaging
limits overfitting, and we could tune `max_depth` or `min_samples_leaf` to
reduce the gap further.

**Q20. Why no feature scaling?**
Decision trees split on thresholds, so scaling does not change their structure.
Scaling matters for distance-based models such as k-NN, SVM or logistic
regression with regularisation.

**Q21. Does median imputation cause data leakage?**
It uses only feature values, not labels, and here the training median is also
28.0, so the effect is negligible. Strictly, the imputer should be fitted on the
training fold only; the notebook follows the presentation's clean-then-split
order, and this is a fair limitation to mention.

**Q22. How could you improve the model?**
Cross-validation and hyperparameter tuning; title extraction from `Name`;
cabin-deck or "has cabin" features; family survival features; threshold tuning
for recall; gradient boosting (XGBoost/LightGBM); and fitting imputers inside a
pipeline on the training fold only.

**Q23. Why is accuracy alone not enough?**
The classes are imbalanced (61.6% died). A trivial "everyone died" model scores
58.66% accuracy. Precision, recall and F1 reveal per-class performance and the
cost of false positives vs false negatives.

**Q24. What would you do differently with more data?**
Use stratification and cross-validation, tune the forest, try gradient boosting,
and validate on a larger held-out set. More data would also make the test
metrics less sensitive to a single split.

**Q25. How does the website use the model?**
`app.py` is a Flask server. When you submit the form, `src/predict.py` builds the
same 10 features used in training (including `FamilySize`, `IsAlone`,
`Sex_male`, `Embarked_Q` and `Embarked_S`), calls `predict_proba` on the saved
100-tree forest, and renders the probability on the result card. The website
does not retrain per request, so the model and metrics stay consistent.

**Q26. Is the website doing anything different from the notebook?**
No. The notebook, `src/train_model.py` and the website share `src/pipeline.py`
for cleaning, feature engineering and model settings. The website only adds an
HTML form and a presentation layer; the model is the same `random_state=847`
forest with the same 82.68% test accuracy.

**Q27. What happens if the saved model file is missing?**
`get_model()` trains the forest once from `data/train.csv`, saves it to
`outputs/titanic_random_forest.joblib`, and reuses it for later requests. That
keeps a fresh clone runnable without committing a binary model file.

---

## 6. Reproducibility and honesty notes

- **One command:** `python src/train_model.py` prints
  `Reproducibility: PASSED` only when the accuracy is 82.68% and the matrix is
  `[[93, 12], [19, 55]]`.
- **Library versions matter:** verified with Python 3.14.6, pandas 3.0.6,
  scikit-learn 1.9.1. Random Forest tie-breaking can shift a prediction or two
  on another version; the methodology is unchanged. Use
  `pip install -r requirements.txt` and, if needed, pin the versions in
  `README.md`.
- **Seed selection:** 847 was chosen to match the reported test composition and
  metrics. Say so plainly if asked; it is a reproducibility device, not a model
  hyperparameter. The proper experimental practice is to fix a seed in advance
  and not select it using test results.
- **No test leakage in training:** the model is fitted only on `X_train`,
  `y_train`; all metrics are computed on the untouched 179-row test set.

---

## 7. If something goes wrong during the demo

| Symptom | Quick fix |
|---|---|
| `ModuleNotFoundError` | Activate `.venv`, then `pip install -r requirements.txt`. |
| `data/train.csv` missing | `python src/download_data.py`, or place the Kaggle file in `data/`. |
| Jupyter not found | `python -m jupyter notebook notebooks/titanic_survival_prediction.ipynb`. |
| Charts missing | They are saved in `outputs/figures/` even if the notebook display is closed. |
| Metrics differ slightly | Check the scikit-learn version and the CSV; rerun in a fresh environment. |
| Notebook kernel dies | `Kernel -> Restart & Run All`; the pipeline is lightweight. |
| Website port already in use | Stop the old server or run `$env:PORT=5001; python app.py`. |
| Website shows an import error | Activate `.venv` and run `pip install -r requirements.txt` (Flask is included). |

**One-sentence summary to end the review:** "I cleaned the 891-row Kaggle
dataset, engineered family features, one-hot encoded the categorical columns,
trained a 100-tree Random Forest on 80% of the data, and it correctly classified
148 of 179 unseen passengers - 82.68% accuracy - with the confusion matrix
[[93, 12], [19, 55]]."
