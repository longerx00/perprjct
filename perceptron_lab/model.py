
import numpy as np
from numpy.typing import NDArray


class Perceptron:
    """Перцептрон для двух классов: 0 и 1"""

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        """Создаёт перцептрон с нулевыми весами"""
        if n_features <= 0:
            raise ValueError("n_features должно быть больше 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate должна быть больше 0")
        if epochs <= 0:
            raise ValueError("epochs должно быть больше 0")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: NDArray[np.float64] = np.zeros(n_features)
        self.bias = 0.0
        self.errors: list[int] = []

    def activate(self, z: float) -> int:
        """Ступенька: 1 если z >= 0, иначе 0"""
        if z >= 0:
            return 1
        return 0

    def predict(self, x: NDArray[np.float64]) -> int:
        """Предсказывает класс точки"""
        z = float(np.dot(x, self.w) + self.bias)
        return self.activate(z)

    def train(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> list[int]:
        """Обучает перцептрон, возвращает ошибки по эпохам"""
        if len(X) == 0:
            raise ValueError("нет данных для обучения")
        if len(X) != len(y):
            raise ValueError("в X и y должно быть одинаковое число точек")
        if len(X[0]) != self.n_features:
            raise ValueError(f"у каждой точки должно быть {self.n_features} числа")

        self.errors = []
        for _epoch in range(self.epochs):
            errors = 0
            for j in range(len(X)):
                x = X[j]
                target = y[j]
                pred = self.predict(x)
                update = self.learning_rate * (target - pred)
                for i in range(self.n_features):
                    self.w[i] = self.w[i] + update * x[i]
                self.bias = self.bias + update
                if update != 0:
                    errors = errors + 1
            self.errors.append(errors)
            if errors == 0:
                break
        return self.errors

    def score(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> float:
        """Доля правильных ответов"""
        if len(X) == 0:
            raise ValueError("нельзя посчитать точность на пустых данных")
        correct = 0
        for j in range(len(X)):
            if self.predict(X[j]) == y[j]:
                correct = correct + 1
        return correct / len(X)
