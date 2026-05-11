from preprocessing import load_and_preprocess
from models import get_models
from evaluation import evaluate_model
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from pathlib import Path
from visualization import (
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_roc_curves,
    plot_threshold_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

models = get_models()
results = []
trained_models = {}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Running {name}...")

    # Hyperparameter tuning
    if name == "Random Forest":
        param_grid = {
            "n_estimators": [50],
            "max_depth": [10]
        }
        model = GridSearchCV(model, param_grid, cv=cv, scoring="f1", n_jobs=-1)

    if name == "Gradient Boosting":
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
            "max_iter": [100]
        }
        model = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    trained_models[name] = model

    results.append({
        "Model": name,
        **metrics
    })

# Save results and visualizations
df_results = pd.DataFrame(results)
print(df_results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_results.to_csv(OUTPUT_DIR / "results.csv", index=False)

plot_metric_comparison(df_results, OUTPUT_DIR)

roc_data = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    plot_confusion_matrix(y_test, y_pred, name, OUTPUT_DIR)
    roc_data[name] = (y_test, y_score)

plot_roc_curves(roc_data, OUTPUT_DIR)

gradient_boosting_scores = roc_data["Gradient Boosting"][1]
threshold_rows = []
for threshold in [i / 100 for i in range(5, 100, 5)]:
    y_pred = (gradient_boosting_scores >= threshold).astype(int)
    threshold_rows.append({
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUTPUT_DIR / "gradient_boosting_thresholds.csv", index=False)
plot_threshold_metrics(threshold_df, "Gradient Boosting", OUTPUT_DIR)
