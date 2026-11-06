"""Здесь создаём точки и загружаем iris"""

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def make_blobs_2d(
    n_samples: int,
    seed: int = 42,
    spread: float = 0.5,
) -> tuple[FloatArray, IntArray]:
    """Генерирует два облака точек вокруг разных центров"""
    if n_samples < 2:
        raise ValueError("точек должно быть хотя бы 2")
    if spread <= 0:
        raise ValueError("разброс должен быть больше 0")
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0
    class0 = rng.normal(loc=[-1.0, -1.0], scale=spread, size=(n0, 2))
    class1 = rng.normal(loc=[1.0, 1.0], scale=spread, size=(n1, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * n0 + [1] * n1)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def load_iris_binary() -> tuple[FloatArray, IntArray]:
    """Загружает iris и оставляет только два класса и два признака"""
    iris = load_iris()
    mask = iris.target < 2
    X = np.asarray(iris.data[mask][:, [0, 2]], dtype=np.float64)
    y = np.asarray(iris.target[mask], dtype=np.int_)
    return X, y


def train_test_split(
    X: FloatArray, y: IntArray, test_size: float = 0.2, seed: int = 42
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Перемешивает данные и делит их на обучающую и тестовую части"""
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть между 0 и 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_test = max(1, int(len(X) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
