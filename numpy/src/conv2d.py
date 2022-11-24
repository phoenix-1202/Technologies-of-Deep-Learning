import numpy as np

from src.activation import ReLU
from src.layer import Layer


class Conv2d(Layer):
    def __init__(self, kernel_size, stride, channels, padding='valid', activation=ReLU()):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_size = None
        self.H, self.W, self.C = None, None, channels
        self.H_prev, self.W_prev, self.C_prev = None, None, None
        self.weights = None
        self.biases = None
        self.activation = activation
        self.cache = {}

    def init(self, in_dim):
        self.padding_size = 0 if self.padding == 'valid' else int((self.kernel_size - 1) / 2)

        self.H_prev, self.W_prev, self.C_prev = in_dim
        self.H = int((self.H_prev - self.kernel_size + 2 * self.padding_size) / self.stride + 1)
        self.W = int((self.W_prev - self.kernel_size + 2 * self.padding_size) / self.stride + 1)

        self.weights = np.random.randn(self.kernel_size, self.kernel_size, self.C_prev, self.C)
        self.biases = np.zeros((1, 1, 1, self.C))

    def forward(self, a_prev, training):
        batch_size = a_prev.shape[0]
        a_prev_padded = Conv2d.zero_pad(a_prev, self.padding_size)
        out = np.zeros((batch_size, self.H, self.W, self.C))

        for i in range(self.H):
            v_start = i * self.stride
            v_end = v_start + self.kernel_size

            for j in range(self.W):
                h_start = j * self.stride
                h_end = h_start + self.kernel_size

                out[:, i, j, :] = np.sum(a_prev_padded[:, v_start:v_end, h_start:h_end, :, np.newaxis] *
                                         self.weights[np.newaxis, :, :, :], axis=(1, 2, 3))

        z = out + self.biases
        a = self.activation.f(z)

        if training:
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        batch_size = da.shape[0]
        a_prev, z, a = (self.cache[key] for key in ('a_prev', 'z', 'a'))
        a_prev_pad = Conv2d.zero_pad(a_prev, self.padding_size) if self.padding_size != 0 else a_prev

        da_prev = np.zeros((batch_size, self.H_prev, self.W_prev, self.C_prev))
        da_prev_pad = Conv2d.zero_pad(da_prev, self.padding_size) if self.padding_size != 0 else da_prev

        dz = da * self.activation.df(z, cached_y=a)
        db = 1 / batch_size * dz.sum(axis=(0, 1, 2))
        dw = np.zeros((self.kernel_size, self.kernel_size, self.C_prev, self.C))

        for i in range(self.H):
            v_start = self.stride * i
            v_end = v_start + self.kernel_size

            for j in range(self.W):
                h_start = self.stride * j
                h_end = h_start + self.kernel_size

                da_prev_pad[:, v_start:v_end, h_start:h_end, :] += \
                    np.sum(self.weights[np.newaxis, :, :, :, :] * dz[:, i:i + 1, j:j + 1, np.newaxis, :], axis=4)

                dw += np.sum(a_prev_pad[:, v_start:v_end, h_start:h_end, :, np.newaxis] *
                             dz[:, i:i+1, j:j+1, np.newaxis, :], axis=0)

        dw /= batch_size

        if self.padding_size != 0:
            da_prev = da_prev_pad[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size, :]

        return da_prev, dw, db

    def get_output_dim(self):
        return self.H, self.W, self.C

    def update_params(self, dw, db):
        self.weights -= dw
        self.biases -= db

    def get_params(self):
        return self.weights, self.biases

    @staticmethod
    def zero_pad(x, pad):
        return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
