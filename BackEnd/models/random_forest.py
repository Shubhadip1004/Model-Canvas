# backend/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier

def build_random_forest(n_estimators: int = 100):
    return RandomForestClassifier(n_estimators=n_estimators)
