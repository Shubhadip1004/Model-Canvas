# backend/models/neural_network.py
from sklearn.neural_network import MLPClassifier

def build_mlp(hidden_layer_sizes = (50,)):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500)
