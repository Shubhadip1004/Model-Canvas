# backend/models/decision_tree.py
from sklearn.tree import DecisionTreeClassifier

def build_decision_tree(max_depth: int = 5):
    return DecisionTreeClassifier(max_depth=max_depth)
