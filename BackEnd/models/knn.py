from sklearn.neighbors import KNeighborsClassifier

def build_knn(n_neighbors: int = 5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)
