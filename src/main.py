from preprocessing import load_and_preprocess
from models import get_models
from evaluation import evaluate_model
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess("data/diabetic_data.csv")

models = get_models()
results = []

for name, model in models.items():
    print(f"Running {name}...")

    # Hyperparameter tuning
    if name == "Random Forest":
        param_grid = {
            "n_estimators": [50],
            "max_depth": [10]
        }
        model = GridSearchCV(model, param_grid, cv=5)

    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    results.append({
        "Model": name,
        **metrics
    })

# Save results
df_results = pd.DataFrame(results)
print(df_results)

df_results.to_csv("results.csv", index=False)