from functools import reduce

import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim, layers, loss_function, optimizer, l2_lambda=0):
        self.layers = layers
        self.weights_grads = {}
        self.biases_grads = {}
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

        self.layers[0].init(input_dim)
        for prev_layer, curr_layer in zip(self.layers, self.layers[1:]):
            curr_layer.init(prev_layer.get_output_dim())

        self.trainable_layers = set(layer for layer in self.layers if layer.get_params() is not None)
        self.optimizer = optimizer(self.trainable_layers)
        self.optimizer.initialize()

    def forward_prop(self, x, training=True):
        a = x
        for layer in self.layers:
            a = layer.forward(a, training)

        return a

    def backward_prop(self, a_last, y):
        da = self.loss_function.grad(a_last, y)
        batch_size = da.shape[0]

        for layer in reversed(self.layers):
            da_prev, dw, db = layer.backward(da)

            if layer in self.trainable_layers:
                if self.l2_lambda != 0:
                    self.weights_grads[layer] = dw + (self.l2_lambda / batch_size) * layer.get_params()[0]
                else:
                    self.weights_grads[layer] = dw

                self.biases_grads[layer] = db

            da = da_prev

    def predict(self, x):
        a_last = self.forward_prop(x, training=False)
        return a_last

    def update_param(self, learning_rate, step):
        self.optimizer.update(learning_rate, self.weights_grads, self.biases_grads, step)

    def compute_loss(self, a_last, y):
        loss = self.loss_function.f(a_last, y)
        if self.l2_lambda != 0:
            batch_size = y.shape[0]
            weights = [layer.get_params()[0] for layer in self.trainable_layers]
            l2_loss = (self.l2_lambda / (2 * batch_size)) * reduce(lambda ws, w: ws + np.sum(np.square(w)), weights, 0)
            return loss + l2_loss
        else:
            return loss

    def train(self, x_train, y_train, mini_batch_size, learning_rate, num_epochs, validation_data):
        x_val, y_val = validation_data
        print(f"Train mode:\n  Batch size is {mini_batch_size}, learning rate is {learning_rate}")
        step = 0
        for e in range(num_epochs):
            print("\nEpoch " + str(e + 1))
            epoch_loss = 0

            if mini_batch_size == x_train.shape[0]:
                mini_batches = (x_train, y_train)
            else:
                mini_batches = NeuralNetwork.create_mini_batches(x_train, y_train, mini_batch_size)

            num_mini_batches = len(mini_batches)
            for i, mini_batch in enumerate(mini_batches, 1):
                mini_batch_x, mini_batch_y = mini_batch
                step += 1
                epoch_loss += self.train_step(mini_batch_x, mini_batch_y, learning_rate, step) / mini_batch_size
                print("\rDone {:1.1%} / 100%".format(i / num_mini_batches), end="")

            print(f"\nTrain loss: {epoch_loss}")

            print("Computing accuracy on test...")
            accuracy = np.sum(np.argmax(self.predict(x_val), axis=1) == y_val) / x_val.shape[0]
            print(f"Accuracy: {accuracy}")

        print("Completed successfully")

    def train_step(self, x_train, y_train, learning_rate, step):
        a_last = self.forward_prop(x_train, training=True)
        self.backward_prop(a_last, y_train)
        loss = self.compute_loss(a_last, y_train)
        self.update_param(learning_rate, step)
        return loss

    @staticmethod
    def create_mini_batches(x, y, mini_batch_size):
        batch_size = x.shape[0]
        mini_batches = []

        p = np.random.permutation(x.shape[0])
        x, y = x[p, :], y[p, :]
        num_complete_minibatches = batch_size // mini_batch_size

        for k in range(0, num_complete_minibatches):
            mini_batches.append((
                x[k * mini_batch_size:(k + 1) * mini_batch_size, :],
                y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            ))

        if batch_size % mini_batch_size != 0:
            mini_batches.append((
                x[num_complete_minibatches * mini_batch_size:, :],
                y[num_complete_minibatches * mini_batch_size:, :]
            ))

        return mini_batches
