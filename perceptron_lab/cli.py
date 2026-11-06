"""Здесь читаем аргументы из терминала и запускаем обучение."""

import argparse
from pathlib import Path

from perceptron_lab.data import (
    FloatArray,
    IntArray,
    load_iris_binary,
    make_blobs_2d,
    train_test_split,
)
from perceptron_lab.model import Perceptron
from perceptron_lab.plots import plot_decision_boundary, plot_errors


def run_experiment(
    name: str, X: FloatArray, y: IntArray, lr: float, epochs: int
) -> None:
    """Обучает перцептрон, печатает точность, сохраняет графики"""
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = Perceptron(n_features=2, learning_rate=lr, epochs=epochs)
    errors = model.train(X_train, y_train)

    print(f"=== {name} ===")
    print(f"эпох понадобилось: {len(errors)} из {epochs}")
    print(f"точность на обучении: {model.score(X_train, y_train):.3f}")
    print(f"точность на тесте: {model.score(X_test, y_test):.3f}")

    folder = Path("plots")
    folder.mkdir(exist_ok=True)
    plot_decision_boundary(model, X, y, folder / f"{name}_boundary.png", title=name)
    plot_errors(errors, folder / f"{name}_errors.png", title=name)


def main(argv: list[str] | None = None) -> int:
    """Главная функция, возвращает 0 если всё хорошо и 1 если ошибка"""
    parser = argparse.ArgumentParser(description="Обучение перцептрона")
    parser.add_argument("--samples", type=int, default=200, help="число точек")
    parser.add_argument("--lr", type=float, default=0.1, help="скорость обучения")
    parser.add_argument("--epochs", type=int, default=50, help="максимум эпох")
    args = parser.parse_args(argv)

    try:
        X, y = make_blobs_2d(args.samples)
        run_experiment("synthetic", X, y, args.lr, args.epochs)

        X_iris, y_iris = load_iris_binary()
        run_experiment("iris", X_iris, y_iris, args.lr, args.epochs)
    except ValueError as error:
        print(f"Ошибка: {error}")
        return 1
    return 0
