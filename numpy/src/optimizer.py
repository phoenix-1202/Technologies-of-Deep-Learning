import numpy as np


class Optimizer:
    def __init__(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def initialize(self):
        raise NotImplementedError

    def update(self, learning_rate, w_grads, b_grads, step):
        raise NotImplementedError


class RAdam(Optimizer):
    def __init__(self, trainable_layers, beta1=0.9, beta2=0.999, epsilon=1e-8):
        Optimizer.__init__(self, trainable_layers)
        self.v = {}
        self.s = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize(self):
        for layer in self.trainable_layers:
            w, b = layer.get_params()
            w_shape = w.shape
            b_shape = b.shape
            self.v[('dw', layer)] = np.zeros(w_shape)
            self.v[('db', layer)] = np.zeros(b_shape)
            self.s[('dw', layer)] = np.zeros(w_shape)
            self.s[('db', layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grads, b_grads, step):
        v_correction_term = 1 - np.power(self.beta1, step)
        s_correction_term = 1 - np.power(self.beta2, step)
        v_corrected = {}
        p_inf = 2 / (1 - self.beta2) - 1

        for layer in self.trainable_layers:
            layer_dw = ('dw', layer)
            layer_db = ('db', layer)

            self.v[layer_dw] = (self.beta1 * self.v[layer_dw] + (1 - self.beta1) * w_grads[layer])
            self.v[layer_db] = (self.beta1 * self.v[layer_db] + (1 - self.beta1) * b_grads[layer])

            v_corrected[layer_dw] = self.v[layer_dw] / v_correction_term
            v_corrected[layer_db] = self.v[layer_db] / v_correction_term

            self.s[layer_dw] = (self.beta2 * self.s[layer_dw] + (1 - self.beta2) * np.square(w_grads[layer]))
            self.s[layer_db] = (self.beta2 * self.s[layer_db] + (1 - self.beta2) * np.square(b_grads[layer]))

            p = p_inf - 2 * step * (1 - s_correction_term) / s_correction_term
            if p > 5:
                l_dw = np.sqrt(s_correction_term / (self.s[layer_dw] + self.epsilon))
                l_db = np.sqrt(s_correction_term / (self.s[layer_db] + self.epsilon))
                r = np.sqrt((p_inf * (p - 4) * (p - 2)) / (p * (p_inf - 4) * (p_inf - 2)))
                dw = learning_rate * v_corrected[layer_dw] * r * l_dw
                db = learning_rate * v_corrected[layer_db] * r * l_db
            else:
                dw = learning_rate * v_corrected[layer_dw]
                db = learning_rate * v_corrected[layer_db]

            layer.update_params(dw, db)

