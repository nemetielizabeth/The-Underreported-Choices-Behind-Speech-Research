import os
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import (StratifiedKFold, GridSearchCV)
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
import matplotlib
matplotlib.use("Agg") # for saving our plots on the cluster
import matplotlib.pyplot as plt
import shap
import argparse

"""
How to run:
- sbatch run-xgboost.sh                 to run the job
- squeue -u enemeti                     to check it's running
- tail -f LOGS/runxgboost_662554.out    to check logs
- scancel jobid                         to cancel

How the script works:
- set the input file in the script
- sbatch run-xgboost.sh --target screening (or --target balanced)
- hyperparameters were selected via ROC-AUC (Screening) or average precision (Balanced) on inner folds
"""

# ------------------------------ Prepare Input ------------------------------ #

# Retrieve master datafile (from prepare-xgboost-input-file.py)
input_csv = "/labs/bozkurtlab/data/Fabla/XGBOOST/From OpenSmile/First_recordings_(not_wordiest)/Raws/dataset_xgboost_P11_raw_cat-y-crop-5min_64p_88f_0demog.csv"
df = pd.read_csv(input_csv)
RUN_NAME = Path(input_csv).stem

# (!) Set depression label cutoff: Screening (balanced dataset) or Balanced (stricter, more imbalanced, fewer +ves)
parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=["screening", "balanced"], required=True,
                    help="Depression label cutoff")
args = parser.parse_args()

# optimize for AUC-ROC on Screening runs, AUC-PRC, better for class imbalance on Balanced runs
target_col = ("IDAS_MDD_Screening_Binary" if args.target == "screening"
              else "IDAS_MDD_Balanced_Binary")
target_str = args.target
scoring_metric = "roc_auc" if target_str == "screening" else "average_precision"

# print our job name just for checking results
print(f"Job name: {RUN_NAME}_{target_str}\n")

# Columns that are NOT features
# prevents label leakage where model accidentally learns from things it shouldn't
non_feature_cols = [
    "ParticipantID",
    "IDAS_MDD_Screening_Binary",
    "IDAS_MDD_Balanced_Binary",
    "IDAS_Total"]

# Create feature matrix X and label vector y by grabbing correct columns
feature_cols = [c for c in df.columns if c not in non_feature_cols]
X = df[feature_cols]    # our input variables the model learns from
y = df[target_col]      # outcome we're trying to predict 

# Check data dist
print("Data shape:", X.shape)
print("Number of +ves:", y.sum(), "out of", len(y))
print("Number of -ves:", (len(y) - y.sum()), "out of", len(y))

# Set plot dir
plot_dir = "/labs/bozkurtlab/data/Fabla/TASKS/LOGS"

# ------------------------------ Define Iterations, Base Model, Hyperparameters ------------------------------ #

all_seed_results = [] # store our metrics across all runs
all_shap_values = []  # store out-of-fold SHAP matrices (one per outer fold, per seed)
all_shap_X = []       # store the matching feature rows for each out-of-fold SHAP matrix
all_y_true = []       # store for confusion matrix
all_y_pred = []       # store for confusion matrix

# set iterations (followed to Mu et al. 2025 for shallow models)
# 42 always produces the same fold split, seed 43 always produces the same fold split, and so on
RUNS = 100
random_state = [42 + i for i in range(RUNS)] # always [42, 43, 44, ...]

for seed in random_state:

    # set base model
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",  
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed)

    # set hyperparameters
    param_grid = {
        "max_depth":     [2, 3, 4],          # tree depth: higher = more complex, risk overfitting
        "n_estimators":  [50, 100, 200],     # number of trees: more = stronger but slower
        "learning_rate": [0.01, 0.05, 0.1]}  # step size: smaller = learns slower but more precise

# ------------------------------ Nested K-Fold CV ------------------------------ #

    """
    FOR each outer fold (1 through 5):
        1. Split data into outer train (80%) and outer test (20%)
        2. Run inner CV + grid search using ONLY outer train
        3. Get winning model
        4. Evaluate winning model on outer test
        5. Save results
    END FOR
    """
    print("\nStarting nested cross-validation")

    # Set up stratified loops
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # outer loop
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)  # inner loop

    outer_results = []

    # split into 5 folds & add enumerate to track which fold we're on
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"\n========== OUTER FOLD {outer_fold} ==========")

        # ----- 1: Split into outer train/test -----

        # get labels & features from correct rows for training/testing
        X_train = X.iloc[train_idx]  # training feature data
        y_train = y.iloc[train_idx]  # training labels
        X_test  = X.iloc[test_idx]   # testing feature data
        y_test  = y.iloc[test_idx]   # testing labels
        print(f"Train: {X_train.shape[0]} participants | Test: {X_test.shape[0]} participants")

        # ----- 2: Run inner CV + grid search (uses only train data) -----

        # set up our grid search WITHIN the outer loop
        grid = GridSearchCV(
            estimator=base_model,         # use model we defined to try variants on
            param_grid=param_grid,        # use list of hyperparams we defined to search over
            scoring=scoring_metric,        
            cv=inner_cv,                  # use 2-fold cv we defined
            refit=True,                   # after finding best hyperparam combo, refit best model on full X_train at the end, this is now grid.best_estimator_
            n_jobs=-1,                    # use all cpus available to me
            verbose=1,                    # for logs
            error_score='raise')          # for logs

        print("Running inner 2-fold CV for hyperparameter tuning...")
        grid.fit(X_train, y_train)        #  runs 27*2 fits + 1 refit on full X_train

        # ----- 3: Get winning model for current outer fold -----
        
        # trained model using the winning hyperparameters
        best_model = grid.best_estimator_ #  this is the refitted model on outer-train

        # hyperparameters that won the inner CV grid search
        best_params = grid.best_params_
        print(f"Winning hyperparameters: {best_params}")

        # mean AUC/PRC (depending on screening vs balanced run) across the 2 inner folds for the winning hyperparameter combo
        validation_score = grid.best_score_
        print(f"Validation {scoring_metric} (mean of 2 inner folds): {validation_score:.3f}")

        # ----- 4: Evaluate current fold on outer test ----
        
        # get continuous probability scores for each test participant
        y_proba = best_model.predict_proba(X_test)[:, 1] #  here we evaluate on outer-test
        
        # convert probabilities to binary predictions (changed to this defalt one)
        y_pred = (y_proba >= 0.5).astype(int)

        # get every participant's true label paired with their prediction for confusion matrix
        all_y_true.extend(y_test.tolist()) # 
        all_y_pred.extend(y_pred.tolist())
        
        # use probabilities to get AUC & PRC
        test_auc_roc = roc_auc_score(y_test, y_proba)
        print(f"TEST AUC ROC: {test_auc_roc:.3f}")

        test_auc_prc = average_precision_score(y_test, y_proba) # switched to prc from auc
        print(f"TEST AUC PRC: {test_auc_prc:.3f}")
        
        # get macro f1
        test_f1_macro = f1_score(y_test, y_pred, average="macro")
        print(f"TEST Macro F1: {test_f1_macro:.3f}")
        
        # get binary f1
        test_f1_binary = f1_score(y_test, y_pred, average="binary")
        print(f"TEST Binary F1: {test_f1_binary:.3f}")

        # get precision
        test_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        print(f"TEST Precision: {test_precision:.3f}")

        # get recall
        test_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        print(f"TEST Recall: {test_recall:.3f}")

        # ----- SHAP on outer-test (out-of-fold) -----
        # Explain the SAME validated model (best_model, trained on outer-train ONLY)
        # on the held-out test rows it never saw. This mirrors how the metrics above
        # are computed (out-of-fold), so the SHAP plot describes the validated model
        # rather than a separate model fit on all the data (which would leak).
        explainer = shap.TreeExplainer(best_model)  # tree explainer for this fold's xgboost model
        shap_test = explainer.shap_values(X_test)   # SHAP for held-out rows only (no leakage)
        all_shap_values.append(shap_test)           # one out-of-fold SHAP matrix per outer fold
        all_shap_X.append(X_test)                   # keep matching feature rows so plot aligns

        # ----- 5: Save -----
        
        # collect all our results from these 5 folds for aggregation next
        outer_results.append({
            "outer_fold": outer_fold,
            "best_params": best_params,
            "validation_score": validation_score,
            "test_auc_prc": test_auc_prc,
            "test_auc_roc": test_auc_roc,
            "test_f1_macro": test_f1_macro,
            "test_f1_binary": test_f1_binary,
            "test_precision": test_precision, 
            "test_recall": test_recall
            })

# ------------------------------ Aggregate K-Folds (per seed) ------------------------------ #

    # grab our 5 fold scores that we saved above for all 6 metrics
    auc_roc_scores = np.array([r["test_auc_roc"] for r in outer_results])
    auc_prc_scores = np.array([r["test_auc_prc"] for r in outer_results]) 
    f1_macro_scores = np.array([r["test_f1_macro"] for r in outer_results])
    f1_b_scores = np.array([r["test_f1_binary"] for r in outer_results])
    precision_scores = np.array([r["test_precision"] for r in outer_results])
    recall_scores = np.array([r["test_recall"] for r in outer_results])

    # calc mean & standard deviations
    print("AUC-ROC: mean = {:.3f}, std = {:.3f}".format(auc_roc_scores.mean(), auc_roc_scores.std()))
    print("AUC-PRC: mean = {:.3f}, std = {:.3f}".format(auc_prc_scores.mean(), auc_prc_scores.std()))
    print("Macro F1: mean = {:.3f}, std = {:.3f}".format(f1_macro_scores.mean(), f1_macro_scores.std()))
    print("Binary F1: mean = {:.3f}, std = {:.3f}".format(f1_b_scores.mean(), f1_b_scores.std()))
    print("Precision: mean = {:.3f}, std = {:.3f}".format(precision_scores.mean(), precision_scores.std()))
    print("Recall: mean = {:.3f}, std = {:.3f}".format(recall_scores.mean(), recall_scores.std()))

    # Store this seed's results for final CI calculation
    all_seed_results.append({
        "seed": seed,
        "mean_auc_roc": auc_roc_scores.mean(),
        "mean_auc_prc": auc_prc_scores.mean(),
        "mean_f1_macro": f1_macro_scores.mean(),
        "mean_f1_binary": f1_b_scores.mean(),
        "mean_precision": precision_scores.mean(),
        "mean_recall": recall_scores.mean()
        })

# ------------------------------ Aggregate Across All Seeds ------------------------------ #

# aggregated across all seeds like Mu et al. 2025 did
# collect the 100 seed-means for each metric
auc_roc_all   = np.array([r["mean_auc_roc"]   for r in all_seed_results])
auc_prc_all   = np.array([r["mean_auc_prc"]   for r in all_seed_results])
f1_macro_all  = np.array([r["mean_f1_macro"]  for r in all_seed_results])
f1_binary_all = np.array([r["mean_f1_binary"] for r in all_seed_results])
precision_all = np.array([r["mean_precision"] for r in all_seed_results])
recall_all    = np.array([r["mean_recall"]    for r in all_seed_results])

# format one metric as: mean [2.5th, 97.5th percentile]
def format_cis(vals):
    return f"{np.mean(vals):.3f} [{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]"

print(f"\n{'-'*60}")
print(f"FINAL RESULTS ACROSS {RUNS} SEEDS (point estimate [95% CI])")
print(f"{'-'*60}")
print(f"AUROC:   {format_cis(auc_roc_all)}")
print(f"AUPRC:   {format_cis(auc_prc_all)}")
print(f"F1 Macro:  {format_cis(f1_macro_all)}")
print(f"F1 Binary: {format_cis(f1_binary_all)}")
print(f"Precision: {format_cis(precision_all)}")
print(f"Recall:    {format_cis(recall_all)}")

# ---------------------------------- Confusion Matrix ------------------------------- #
"""
CM is for reference only, NOT for table transcription - why?
It's pooled across all 100 seeds: each participant appears 100 times, so the support counts
are inflated 100x. So these numbers will differ from the fold-mean estimates above. 
They're just caculated differently. 
"""

cm = confusion_matrix(all_y_true, all_y_pred)
print("\nAggregated Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(all_y_true, all_y_pred, digits=3))

# ------------------------------ SHAP Interpretation ------------------------------ #

"""
SHAP (out-of-fold, all folds and seeds pooled into ONE plot)
Each participant lands in exactly one outer-test fold per seed, so they contribute
one SHAP attribution per seed. We POOL (stack) all those rows rather than averaging
matrices, because the test subsets differ across folds and no longer align row-for-row.
The summary plot reads the distribution of attributions per feature, so a participant
appearing once per seed is expected and fine.
"""
shap_all = np.vstack(all_shap_values)  # stack every out-of-fold SHAP matrix into one tall matrix
X_all = pd.concat(all_shap_X, axis=0)  # stack the matching feature rows in the same order

shap.summary_plot(shap_all, X_all, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig(f"{plot_dir}/{RUN_NAME}_{target_str}_shap_summary_oof.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved to {plot_dir}/{RUN_NAME}_{target_str}_shap_summary_oof.png")
