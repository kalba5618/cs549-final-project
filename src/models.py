from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=True,
            random_state=42
        )
    }
