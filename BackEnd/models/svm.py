# backend/models/svm.py
from typing import Literal
from sklearn.svm import SVC

def build_svm(kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'rbf', C: float = 1.0):
    return SVC(kernel=kernel, C=C, probability=False)
