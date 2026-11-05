"""Графики: граница решения и ошибки по эпохам"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from perceptron_lab.data import FloatArray, IntArray
from perceptron_lab.model import Perceptron


def plot_decision_boundary(
    model: Perceptron,
    X: FloatArray,
    y: IntArray,
    path: Path,
    title: str = "Граница решения",
    xlabel: str = "x1",
    ylabel: str = "x2",
) -> None:
    """Рисует точки и границу решения"""
    if model.n_features != 2:
        raise ValueError("границу можно нарисовать только для 2 признаков")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", label="класс 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", label="класс 1")

    w1, w2 = model.w
    b = model.bias
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    if w2 != 0:
        xs = np.linspace(x_min, x_max, 100)
        ys = -(w1 * xs + b) / w2
        ax.plot(xs, ys, "k--", label="граница")
    elif w1 != 0:
        ax.axvline(-b / w1, color="k", linestyle="--", label="граница")

    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_xlim(x_min, x_max)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_errors(errors: list[int], path: Path, title: str = "Ошибки") -> None:
    """Рисует график ошибок по эпохам"""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(errors) + 1)
    ax.plot(epochs, errors, marker="o")
    ax.set_title(title)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("число ошибок")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    fig.savefig(path)
    plt.close(fig)
