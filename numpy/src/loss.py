import numpy as np

epsilon = 1e-20


class LossFunction:
    def f(self, a_last, y):
        raise NotImplementedError

    def grad(self, a_last, y):
        raise NotImplementedError


class SoftmaxCrossEntropy(LossFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        loss = -1 / batch_size * (y * np.log(np.clip(a_last, epsilon, 1.0))).sum()
        return loss

    def grad(self, a_last, y):
        return - np.divide(y, np.clip(a_last, epsilon, 1.0))

