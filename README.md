# Diabetes Prediction — Pima Indians Dataset

## Problem

Predict whether a patient has diabetes (`Outcome`: 1 = diabetic, 0 = not) from
routine clinical measurements, and compare three classifiers — a C5.0 decision
tree (including a boosted, cross-validation-tuned version), K-nearest
neighbours, and logistic regression.

## Data source

The Pima Indians Diabetes dataset (768 patients, 8 predictors). The data file
is **not included in this repository** — download it as `diabetes.csv` and
place it next to `finalproject_code_.R` before running:

- UCI Machine Learning Repository — Pima Indians Diabetes dataset:
  https://archive.ics.uci.edu/dataset/34/diabetes
- Kaggle mirror ("Pima Indians Diabetes Database"), which provides the
  `diabetes.csv` file with the exact column names this script expects:
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Expected columns: `Pregnancies`, `Glucose`, `BloodPressure`,
`SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`,
`Outcome`.

**Data caveat handled in the script:** in this dataset, a value of 0 for
`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, or `BMI` is
physiologically impossible and encodes a missing measurement. The script
converts those zeros to `NA` and imputes them with medians computed on the
training split only.

## Methods

Implemented in `finalproject_code_.R`:

- Stratified 70/30 train/test split (`caret::createDataPartition`, so class
  proportions are preserved).
- All preprocessing is learned from the training split only: median
  imputation (above), and standardisation using training-set means/SDs
  applied to the test set. KNN uses the scaled data.
- **No tuning on the test set.** The decision-tree boosting parameters and
  the KNN neighbourhood size `k` are chosen by repeated cross-validation on
  the training data; the logistic-regression probability cutoff is chosen
  from out-of-fold (10-fold CV) predictions on the training data. The test
  set is then scored once per model.
- Sensitivity and specificity are computed in code from the confusion
  counts, for every model.
- Logistic regression is additionally evaluated with AUC and a 95%
  confidence interval (`pROC::ci.auc`). For every model, the script
  computes test accuracy minus the **majority-class baseline** (about
  65.1% on the full data — predicting "not diabetic" for everyone), so
  the improvement over the baseline is a printed result, not an eyeball
  comparison.
- A stability check at the end repeats the pipeline over several seeds
  as fully independent replications — each seed makes its own split,
  learns its own imputation/scaling statistics, and re-tunes its own
  `k` and cutoff on its own training data — and reports the mean and
  standard deviation of the test metrics across splits.

## Results

Run the script to produce the current numbers — it prints, for each model,
test-set accuracy, sensitivity, and specificity; the logistic-regression
AUC with its 95% CI; and the cross-seed stability summary. Any accuracy
claim for these models is only meaningful relative to the 65.1%
majority-class baseline printed by the script. (No result figures are
quoted here because they depend on the downloaded data file and the
package versions used; the script is the source of truth.)

## How to run

```bash
# 1. download diabetes.csv (see Data source) into this directory
# 2. install the R packages listed in requirements.txt
# 3. run:
Rscript finalproject_code_.R
```

## Coursework note

This is a coursework group project (predictive analytics class project —
see `Final Project_Diabetes_Group2.pdf` and `Predictive Analytics.pptx` in
this repository), not production or clinical software. Its predictions must
not be used for medical decisions.
