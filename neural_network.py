import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nn_utils
import pandas as pd


class NeuralNetworkNew:
    def __init__(self, layers, net_type="classification", dropout=None):
        self.layers = layers
        self.L = len(layers)
        self.net_type = net_type
        self.dropout = dropout
        self.num_features = layers[0]
        self.num_classes = layers[-1]

        self.W = {}
        self.b = {}

        self.dW = {}
        self.db = {}

        self.setup()

    def setup(self):

        # Initalize weights
        for i in range(1, self.L):
            # Xavier Initialziation, it does help with convergance
            self.W[i] = tf.Variable(
                tf.random.normal(shape=(self.layers[i], self.layers[i - 1]))
                * np.sqrt(2 / self.layers[0])
            )
            # self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1]))) #Wx+b
            self.b[i] = tf.Variable(tf.ones(shape=(self.layers[i], 1)))  # b

    def forward_pass(self, X):
        A = tf.convert_to_tensor(
            X, dtype=tf.float32
        )  # Get input to work with tf, important to keep type consistent
        for i in range(1, self.L):
            if self.dropout is not None:
                pass
            # Applying weights
            Z = tf.matmul(A, tf.transpose(self.W[i])) + tf.transpose(self.b[i])
            # activation
            if i != self.L - 1:
                A = tf.nn.relu(Z)
            else:
                # final layer activation, in this case nothing happens
                A = Z
        return A

    def compute_loss(self, A, Y):
        if self.net_type == "classification":
            loss = tf.nn.softmax_cross_entropy_with_logits(Y, A)
        elif self.net_type == "regression":
            mse = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            )
            loss = mse(Y, A)
        return tf.reduce_mean(loss)
        # return loss

    def update_params(self, lr):
        for i in range(1, self.L):
            self.W[i].assign_sub(lr * self.dW[i])
            self.b[i].assign_sub(lr * self.db[i])

    def predict(self, X):
        if self.net_type == "classification":
            A = self.forward_pass(X)
            return tf.argmax(tf.nn.softmax(A), axis=1)
        elif self.net_type == "regression":
            return self.forward_pass(X)

    def info(self):
        num_params = 0
        for i in range(1, self.L):
            num_params += self.W[i].shape[0] * self.W[i].shape[1]
            num_params += self.b[i].shape[0]
        print(f"Type: {self.net_type}")
        print("Input Features:", self.num_features)
        if self.net_type == "classification":
            print("Number of Classes:", self.num_classes)
        print("Hidden Layers:")
        print("--------------")
        for i in range(1, self.L - 1):
            print(f"Layer {i}, Units {self.layers[i]}")
        print("--------------")
        print("Number of parameters:", num_params)

    def train_on_batch(self, X, Y, lr):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            A = self.forward_pass(X)
            loss = self.compute_loss(A, Y)

        for i in range(1, self.L):
            self.dW[i] = tape.gradient(loss, self.W[i])
            self.db[i] = tape.gradient(loss, self.b[i])
        del tape
        self.update_params(lr)
        return loss.numpy()

    def metric(self, preds, y):
        if self.net_type == "classification":
            return np.mean(preds.numpy() == np.argmax(y, axis=1))
        else:
            # print(preds)
            return np.square(preds.numpy() - y).mean()

    def train(
        self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr
    ):
        # Your code here
        history = {"val_loss": [], "train_loss": [], "val_metrics": []}

        for e in range(epochs):
            epoch_train_loss = 0
            print(f"Epoch {e}", end=".")
            for i in range(steps_per_epoch):
                x_batch = x_train[i * batch_size : (i + 1) * batch_size]
                y_batch = y_train[i * batch_size : (i + 1) * batch_size]

                batch_loss = self.train_on_batch(x_batch, y_batch, lr)
                epoch_train_loss += batch_loss

                if i % int(np.ceil(steps_per_epoch / 10)) == 0:
                    print(end=".")

            history["train_loss"] = epoch_train_loss
            val_A = self.forward_pass(x_test)
            val_loss = self.compute_loss(val_A, y_test).numpy()
            history["val_loss"] = val_loss
            val_preds = self.predict(x_test)
            val_metric = self.metric(val_preds, y_test)
            history["val_metric"] = val_metric
            print("val metric:", val_metric)
        return history
