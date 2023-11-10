import numpy as np
from sklearn.metrics import mean_squared_error


def calc_acc(label, pred):
    return np.mean(label == pred) if len(label) == len(pred) else -1


def calc_mse_from_cls_labels(label, pred, val_map):
    val_map_func = np.vectorize(dict(val_map).get)
    y_true, y_pred = val_map_func(label), val_map_func(pred)
    return mean_squared_error(y_true, y_pred)


def calc_prediction_distribution(pred, class_names):
    unique, counts = np.unique(pred, return_counts=True)
    res = {val: cnt / len(pred)
           for val, cnt in zip(unique, counts)}
    # Set non-predicted labels to 0 frequency
    for cls in class_names:
        res[cls] = res.get(cls, 0)
    return res
