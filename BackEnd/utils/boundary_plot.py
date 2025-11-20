# backend/utils/boundary_plot.py
import numpy as np

"""
make_decision_boundary_grid(model, X, padding=0.5, step=0.05)
- returns flattened grid xx, yy and preds for plotting as a contour/heatmap
"""
def make_decision_boundary_grid(model, X, padding=0.5, step=0.05):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx_vals = np.arange(x_min, x_max, step)
    yy_vals = np.arange(y_min, y_max, step)
    xx, yy = np.meshgrid(xx_vals, yy_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model.predict(grid_points)
    except Exception:
        # some models expect shape etc. try with numpy array
        Z = np.array([model.predict(pt.reshape(1, -1))[0] for pt in grid_points])
    return {
        "xx": xx.ravel(),
        "yy": yy.ravel(),
        "preds": Z.ravel(),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "nx": len(xx_vals),
        "ny": len(yy_vals)
    }
