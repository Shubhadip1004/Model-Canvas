
from sklearn.linear_model import LogisticRegression

def build_logistic(max_iter: int = 500):
    return LogisticRegression(max_iter=max_iter)
