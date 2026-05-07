from preprocessing import load_and_preprocess
from models import get_models
from evaluation import evaluate_model
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

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

# Save results
df_results = pd.DataFrame(results)
print(df_results)

df_results.to_csv("results.csv", index=False)

# Create folder for visualizations
FIGURES_DIR.mkdir(exist_ok=True)

# Save confusion matrix for each model
for name, model in trained_models.items():
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# Save ROC curve comparison
plt.figure()

for name, model in trained_models.items():
    if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

plt.title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curve_comparison.png")
plt.close()

# Save bar chart comparing model metrics
metric_cols = ["accuracy", "precision", "recall", "f1"]
df_plot = df_results.set_index("Model")[metric_cols]

df_plot.plot(kind="bar", figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_performance_comparison.png")
plt.close()
