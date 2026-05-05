from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50)
    }