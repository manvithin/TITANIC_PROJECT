# Dataset

`train.csv` is the standard training file from the Kaggle competition
**Titanic: Machine Learning from Disaster**:

- Competition: <https://www.kaggle.com/c/titanic>
- File: `train.csv`
- Shape: **891 rows x 12 columns**

| Column | Meaning |
|---|---|
| `PassengerId` | Row identifier (not a model feature) |
| `Survived` | Target: 0 = died, 1 = survived |
| `Pclass` | Passenger class (1, 2, 3) |
| `Name` | Passenger name (not used as a raw feature) |
| `Sex` | male / female |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number (not used as a raw feature) |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number (dropped: 687 missing) |
| `Embarked` | Port: C = Cherbourg, Q = Queenstown, S = Southampton |

## Expected missing values

| Column | Missing |
|---|---:|
| `Cabin` | 687 |
| `Age` | 177 |
| `Embarked` | 2 |

The script in `src/train_model.py` verifies these counts and prints them.

## How this copy was obtained

The file was downloaded from the public mirror
<https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv>,
which is a copy of the standard Kaggle `train.csv`.

If `train.csv` is ever removed, restore it with either:

```bash
python src/download_data.py --force
```

or download it manually from <https://www.kaggle.com/c/titanic/data> and save it
at `data/train.csv`.

The data is publicly available competition data, intended for learning and
demonstration. Refer to Kaggle's competition rules for any reuse.
