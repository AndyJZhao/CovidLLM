import numpy as np


def calc_acc(label, pred):
    return np.mean(label == pred) if len(label) == len(pred) else -1


def calc_mse_from_cls_labels(label, pred, val_map):
    return np.mean(label == pred) if len(label) == len(pred) else -1
