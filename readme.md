# Classifier Evaluation

`evaluation.py` trains six classification algorithms on a CSV dataset and prints
their Accuracy, Precision, Recall, F1-score, and AU-ROC. It performs one
stratified 80/20 train/test split and does not run generalization tests or create
charts.

## Requirements

Install Python 3 and the required packages:

```bash
python3 -m pip install numpy pandas scikit-learn
```

## Dataset format

Provide a CSV with a `label` column and all 48 feature columns used by the
script. The labels must be binary values `0` and `1`; the metrics treat `1` as
the positive class. Rows with missing or infinite feature values are filled
with zero.

The CSV URL must be directly readable by pandas and accessible from the machine
running the script.

## Run

```bash
python3 evaluation.py
```

When prompted, paste the CSV URL. The script prints a results table for:

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting
- K-Nearest Neighbors (KNN)
- Naive Bayes

Each algorithm is trained on the training portion of the split and evaluated
against the held-out test portion. AU-ROC is calculated from predicted class
probabilities.