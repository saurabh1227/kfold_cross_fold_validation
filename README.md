# K-Fold Cross Validation in Python (scikit-learn)

A hands-on Jupyter notebook that explains **Hold-Out vs K-Fold vs Leave-One-Out cross validation** using a simple fruits classification dataset, with a Decision Tree classifier.

> Part of the hands-on ML material by **[Dataloves](mailto:ask.dataloves@gmail.com)**: GenAI & Data Science Training Institute.

---

## What you will learn

- Why a single train/test split (**Hold-Out**) can give unreliable results
- How **K-Fold Cross Validation** works, and how `KFold` splits the data
- Effect of `shuffle=True` on the splits
- Using `iloc` with fold indices to build train/test sets
- Training a **Decision Tree** (`criterion='entropy'`) inside a CV loop
- Evaluating each fold with **Accuracy, Cohen's Kappa and Recall**, then averaging
- **Leave-One-Out CV (LOOCV)**: the special case where `k = n`

## Notebook outline

| Section | Description |
|---|---|
| Imports | NumPy, Pandas, Matplotlib, Seaborn, scikit-learn |
| Hold-Out method | Two random `train_test_split` calls, and checking that the test sets overlap |
| K-Fold (Case 1) | `KFold(n_splits=4)` without shuffle |
| K-Fold (Case 2) | `KFold(n_splits=4, shuffle=True)` |
| K-Fold (Case 3) | `KFold(n_splits=5, shuffle=True)` on the full DataFrame |
| Fold-wise train/test shapes | Using `iloc` to create each split |
| Model evaluation | Decision Tree + accuracy, kappa and recall per fold, and their mean |
| Leave-One-Out | Theory and note: `LeaveOneOut()` is equivalent to `KFold(n_splits=n)` |

## Dataset

The notebook uses a `fruits.csv` file with these columns:

| Column | Type | Description |
|---|---|---|
| `sphericity` | float | How round the fruit is (feature) |
| `weight` | float | Weight of the fruit (feature) |
| `label` | string | Fruit class, e.g. `Apple` (target) |

> The dataset is **not included** in this repo. Put your `fruits.csv` in a `data/` folder and update the path in the notebook:
>
> ```python
> fruits = pd.read_csv("data/fruits.csv")
> ```

## Getting started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook kfold_cross_validation_231126.ipynb
```

## Quick concept recap

**K-Fold CV**: split the data into `k` equal folds. Train on `k-1` folds, test on the remaining one, repeat `k` times so every fold is the test set once, then average the scores.

**Leave-One-Out CV (LOOCV)**: K-Fold with `k = n`. Each single sample is the test set once, so the model is trained and tested `n` times.

| Method | Pros | Cons |
|---|---|---|
| Hold-Out | Fast and simple | Result depends on one random split |
| K-Fold | More reliable estimate, uses all data | `k` times more training |
| LOOCV | Least bias, uses almost all data to train | Very slow on large datasets |

## Tech stack

Python • NumPy • Pandas • Matplotlib • Seaborn • scikit-learn • Jupyter

## Contact

**Dataloves**
- Phone: +91 91404 97404
- Email: ask.dataloves@gmail.com

---

If this notebook helped you, give the repo a ⭐
