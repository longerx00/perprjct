"""Здесь тесты"""

import numpy as np

from perceptron_lab.cli import main
from perceptron_lab.data import load_iris_binary, make_blobs_2d
from perceptron_lab.model import Perceptron


def test_activate() -> None:
    """Ступенька: 1 при z >= 0 и 0 при z < 0"""
    p = Perceptron(n_features=2)
    assert p.activate(0) == 1
    assert p.activate(-1) == 0


def test_weights_length() -> None:
    """Веса - массив нужной длины"""
    p = Perceptron(n_features=5)
    assert len(p.w) == 5


def test_predict() -> None:
    """При весах (1, 1) точка сверху - класс 1, снизу - класс 0"""
    p = Perceptron(n_features=2)
    p.w = np.array([1.0, 1.0])
    assert p.predict(np.array([2.0, 3.0])) == 1
    assert p.predict(np.array([-2.0, -3.0])) == 0


def test_train_blobs() -> None:
    """После обучения на двух облаках точность 1.0"""
    X, y = make_blobs_2d(200)
    p = Perceptron(n_features=2, epochs=100)
    p.train(X, y)
    assert p.score(X, y) == 1.0


def test_train_iris() -> None:
    """На iris точность тоже 1.0"""
    X, y = load_iris_binary()
    p = Perceptron(n_features=2, epochs=100)
    p.train(X, y)
    assert p.score(X, y) == 1.0


def test_cli_bad_lr() -> None:
    """Отрицательная скорость обучения - понятная ошибка, код 1"""
    assert main(["--lr", "-1"]) == 1
