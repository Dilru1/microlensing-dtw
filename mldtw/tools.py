"""Small helpers shared by the notebooks."""
import numpy as np


def centre_on_peak(X):
    """Shift each light curve so its brightest epoch sits in the middle of the window.

    This removes the t0 (time of peak) nuisance without stretching time,
    so the event duration tE is preserved.
    """
    mid = X.shape[1] // 2
    return np.array([np.roll(x, mid - int(np.argmax(x))) for x in X])


def one_nn_predict(D, y_train):
    """1-nearest-neighbour labels from a (n_test, n_train) distance matrix."""
    return y_train[np.argmin(D, axis=1)]


def knn_regress(D, target_train, k=5):
    """k-nearest-neighbour regression (mean of the k closest training targets)."""
    nn = np.argsort(D, axis=1)[:, :k]
    return target_train[nn].mean(axis=1)


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    return cm
