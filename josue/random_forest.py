"""
Random Forest model for Diabetes Hospital Readmission Prediction.

Usage:
    python random_forest.py
    python random_forest.py --data ../data/diabetic_data.csv --no-smote
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from preprocessing import load_and_preprocess

warnings.filterwarnings("ignore")

# output directory for all results and visualizations
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# hyperparameter tuning function
def tune_random_forest(X_train, y_train, random_state=42):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.
    """
    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
        "class_weight":      ["balanced"],
    }

    # n_jobs=-1 on the RF itself (tree-level parallelism, stable across Python versions)
    # n_jobs=1  on RandomizedSearchCV to avoid joblib/sklearn.parallel conflicts on Python 3.14+
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )

    print("hyperparameter tuning with RandomizedSearchCV")
    t0 = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - t0

    print(f"\nBest parameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV F1 score : {search.best_score_:.4f}")
    print(f"Tuning wall time : {tuning_time:.1f}s")

    # Save CV results for reference
    cv_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    cv_results.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
    print(f"Full CV results saved → outputs/cv_results.csv")

    return search.best_estimator_, search.best_params_, tuning_time


# model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train the final model and evaluate it on the held-out test set.
    """
    # Training time
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Inference time and predictions
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":     accuracy_score(y_test, y_pred),
        "precision":    precision_score(y_test, y_pred, zero_division=0),
        "recall":       recall_score(y_test, y_pred, zero_division=0),
        "f1":           f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":      roc_auc_score(y_test, y_prob),
        "train_time_s": round(train_time, 4),
        "pred_time_s":  round(pred_time, 4),
    }

    print("Random Forest Test Set Performance:")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Train time: {metrics['train_time_s']}s")
    print(f"  Pred time : {metrics['pred_time_s']}s")
    print()
    print("Per-class classification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Readmitted (<30d)", "Readmitted (<30d)"]))

    return metrics, y_pred, y_prob


# visualization functions
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Readmitted", "Readmitted <30d"],
        yticklabels=["Not Readmitted", "Readmitted <30d"],
    )
    plt.title("Random Forest — Confusion Matrix", fontsize=13, pad=12)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → outputs/confusion_matrix.png")


def plot_roc_curve(y_test, y_prob, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", lw=2,
             label=f"Random Forest (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest — ROC Curve", fontsize=13, pad=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → outputs/roc_curve.png")


def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(range(top_n), top_scores, color="blue", edgecolor="white")
    plt.xticks(range(top_n), top_features, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean Decrease in Impurity")
    plt.title(f"Random Forest — Top {top_n} Feature Importances", fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importances.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → outputs/feature_importances.png")

    # save to CSV for use in report tables
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
    print(f"Saved → outputs/feature_importances.csv")



# main function to run the entire pipeline
def main(data_path, use_smote=True, random_state=42):
    print("Random Forest for Diabetes Readmission Prediction")

    # 1. Preprocess 
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        path=data_path,
        random_state=random_state,
        apply_smote=use_smote,
        verbose=True,
    )
    print(f"\nFinal feature count: {len(feature_names)}")

    # 2. Hyperparameter tuning 
    best_model, best_params, tuning_time = tune_random_forest(
        X_train, y_train, random_state=random_state
    )

    # 3. Evaluate on test set 
    metrics, y_pred, y_prob = evaluate_model(
        best_model, X_train, X_test, y_train, y_test
    )

    # 4. Visualizations 
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob, metrics["roc_auc"])
    plot_feature_importance(best_model, feature_names, top_n=20)

    # 5. Save metrics summary 
    summary = {
        "Model": "Random Forest",
        **metrics,
        "tuning_time_s": round(tuning_time, 2),
        "best_params": str(best_params),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "rf_metrics.csv"), index=False)
    print(f"\nMetrics summary saved → outputs/rf_metrics.csv")

    print("Done. All outputs saved to josue/outputs/")

    return metrics, best_model, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest for Diabetes Readmission")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "../data/diabetic_data.csv"),
        help="Path to diabetic_data.csv",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE (use class_weight='balanced' only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(
        data_path=args.data,
        use_smote=not args.no_smote,
        random_state=args.seed,
    )
