from preprocessing import load_and_preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from evaluation import evaluate_model

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay

data, feature_names = load_and_preprocess("data/diabetic_data.csv", return_feature_names=True) 

X_train, X_test, y_train, y_test = data

baseline = DecisionTreeClassifier(random_state=42)
baseline_metrics = evaluate_model(baseline, X_train, X_test, y_train, y_test)

# parameters to test 
params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7],
    "min_samples_split": [10,20],
    "min_samples_leaf": [5,10],
    "class_weight": [None, "balanced"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)

tuned_metrics = evaluate_model(grid, X_train, X_test, y_train, y_test)

#compare baseline vs tuned trees
print("Baseline Decision Tree:")
print(baseline_metrics)

print("\nTuned Decision Tree:")
print(tuned_metrics)

print("\nBest parameters:")
print(grid.best_params_)

best_tree = grid.best_estimator_

plt.figure(figsize=(24, 12))
plot_tree(
    best_tree,
    feature_names=feature_names,
    class_names=["Not Readmitted", "Readmitted <30"],
    filled=True,
    rounded=True,
    max_depth=3 #truncate for ease of reading
)

plt.savefig("dt_visualization.png", dpi=300, bbox_inches="tight")

#confusion matrix
y_pred = best_tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.savefig("dt_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


#feature importances
importances = best_tree.feature_importances_

imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
imp_df.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top Feature Importances (Decision Tree)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("dt_feature_importances.png", dpi=300, bbox_inches="tight")
plt.show()


#precision recall display
PrecisionRecallDisplay.from_estimator(best_tree, X_test, y_test)
plt.title("Precision-Recall Curve")
plt.savefig("dt_precision-recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()
