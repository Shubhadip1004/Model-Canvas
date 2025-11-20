import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
load_dataset_2d(name)
- loads one of iris, wine, breast_cancer, diabetes
- scales features, reduces to 2D (PCA)
- returns (X_train, X_test, y_train, y_test, classes, feature_names)
"""

def load_dataset_2d(name: str):
    name = name.lower()

    # ---- Load dataset safely ----
    if name == "iris":
        ds = datasets.load_iris()
    elif name == "wine":
        ds = datasets.load_wine()
    elif name in ("breast_cancer", "breast-cancer", "cancer"):
        ds = datasets.load_breast_cancer()
    elif name == "diabetes":
        ds = datasets.load_diabetes()
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    # ---- Extract X, y ALWAYS AS NUMPY ARRAYS ----
    if isinstance(ds, tuple) and len(ds) == 2:
        # (data, target)
        X = np.asarray(ds[0])
        y = np.asarray(ds[1])
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        classes = np.unique(y).astype(str)
    else:
        # sklearn Bunch
        X = np.asarray(ds.data)
        y = np.asarray(ds.target)

        feature_names = list(getattr(ds, "feature_names", [f"f{i}" for i in range(X.shape[1])]))
        classes = np.asarray(getattr(ds, "target_names", np.unique(y).astype(str)), dtype=str)

    # ---- For regression dataset like diabetes: convert to binary ----
    if name == "diabetes":
        median = np.median(y)
        y = (y > median).astype(int)
        classes = np.array(["low", "high"])

    # ---- Standardize + PCA ----
    X_scaled = StandardScaler().fit_transform(X)
    X_2d = PCA(n_components=2).fit_transform(X_scaled)

    # ---- Train/Test split ----
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.2, random_state=42, stratify=strat
    )

    return X_train, X_test, y_train, y_test, classes, feature_names