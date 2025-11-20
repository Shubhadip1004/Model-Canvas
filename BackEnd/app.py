# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os

from utils.data_loader import load_dataset_2d
from utils.boundary_plot import make_decision_boundary_grid

from models.knn import build_knn
from models.svm import build_svm
from models.logistic import build_logistic
from models.decision_tree import build_decision_tree
from models.random_forest import build_random_forest
from models.neural_network import build_mlp

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

AVAILABLE_DATASETS = ["iris", "wine", "breast_cancer", "diabetes"]
AVAILABLE_ALGOS = {
    "knn": build_knn,
    "svm": build_svm,
    "logistic": build_logistic,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
    "neural_network": build_mlp
}

@app.get('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.get('/datasets')
def datasets():
    return jsonify({"datasets": AVAILABLE_DATASETS})

@app.get('/algos')
def algos():
    return jsonify({"algorithms": list(AVAILABLE_ALGOS.keys())})

@app.get('/train')
def train():
    # params
    dataset = request.args.get('dataset', 'iris')
    algo = request.args.get('algo', 'logistic')

    if dataset not in AVAILABLE_DATASETS:
        return jsonify({"error": f"Unknown dataset {dataset}"}), 400
    if algo not in AVAILABLE_ALGOS:
        return jsonify({"error": f"Unknown algorithm {algo}"}), 400

    # load 2D data
    (X_train, X_test, y_train, y_test, classes, full_feature_names) = load_dataset_2d(dataset)

    # build model
    builder = AVAILABLE_ALGOS[algo]
    model = builder()  # returns a fitted sklearn-like estimator
    model.fit(X_train, y_train)

    # predictions & metrics
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = float(accuracy_score(y_test, y_pred))

    # make decision boundary grid (predict over grid)
    grid = make_decision_boundary_grid(model, X_train, padding=0.5, step=0.05)

    # prepare response
    resp = {
        "dataset": dataset,
        "algo": algo,
        "accuracy": round(acc, 4),
        "classes": [str(c) for c in classes],
        "feature_names": full_feature_names,
        "train": {
            "x": X_train[:, 0].tolist(),
            "y": X_train[:, 1].tolist(),
            "labels": y_train.tolist()
        },
        "test": {
            "x": X_test[:, 0].tolist(),
            "y": X_test[:, 1].tolist(),
            "labels": y_test.tolist(),
            "preds": y_pred.tolist()
        },
        "grid": {
            "xx": grid["xx"].tolist(),  # flattened
            "yy": grid["yy"].tolist(),  # flattened
            "preds": grid["preds"].tolist(),  # flattened class preds
            "x_min": float(grid["x_min"]),
            "x_max": float(grid["x_max"]),
            "y_min": float(grid["y_min"]),
            "y_max": float(grid["y_max"]),
            "nx": int(grid["nx"]),
            "ny": int(grid["ny"])
        }
    }
    return jsonify(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
