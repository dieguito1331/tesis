# -*- coding: utf-8 -*-

import datetime

import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.callbacks import Callback
from keras.optimizers import SGD


def relu(x):
    return T.maximum(0, x)


class Log_callback(Callback):
    '''Callback that records events'''
    def __init__(self, verbose):
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.best = float("inf")

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.loss = 0
        self.start = datetime.datetime.now()

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        self.loss += logs.get('loss', 0) * batch_size

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        if epoch % 30 == 0:
            print("\nepoch\ttrain_loss\tval_loss\tis_best\ttime")
            print("--------------------------------------------------------------")
        is_best = False
        if logs.get('val_loss', float("inf")) < self.best:
            self.best = logs.get('val_loss', float("inf"))
            is_best = True
        print("{0}\t{1:.5f}\t\t{2:.5f}\t{3}\t{4:.2f} secs"
              .format(epoch,
                      self.loss / self.seen,
                      logs.get('val_loss', float("inf")),
                      is_best,
                      (datetime.datetime.now() - self.start).total_seconds()))


class BinaryMLP(BaseEstimator, ClassifierMixin):

    def __init__(self, hidden=(100, 50), activations=("tanh", "tanh"),
                 drop=(0.5, 0.5), optimizer="sgd", early_stop=True,
                 max_epoch=5000, patience=50, learning_rate=0.01,
                 batch_size=500, verbose=2, save_path=None, loss="binary_crossentropy"):

        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.hidden = hidden
        self.activations = activations
        self.drop = drop
        self.patience = patience
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.activations = activations
        self.batch_size = batch_size
        self.save_path = save_path
        self.optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.best_iteration = 1
        self.loss = loss

    def fit(self, X, y, eval_set=[None], early_stopping_rounds=None, **kwargs):
        if early_stopping_rounds is not None:
            self.patience = early_stopping_rounds

        self.model = Sequential()

        for i, layer in enumerate(self.hidden):
            if layer == 0:
                continue

            activation = self.activations[i]

            if (activation == "relu") and not hasattr(T.nnet, "relu"):
                activation = relu

            params = dict(output_dim=layer,
                          init='glorot_uniform',
                          activation=activation)
            if i == 0:
                params["input_dim"] = X.shape[1]

            self.model.add(Dense(**params))
            self.model.add(Dropout(self.drop[i]))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer)

        if self.early_stop:
            callbacks = [EarlyStopping(monitor='val_loss',
                                       patience=self.patience,
                                       verbose=0)]
            verbose = 0
        else:
            callbacks = []
            verbose = self.verbose

        if self.save_path is not None:
            callbacks.append(ModelCheckpoint(self.save_path,
                                             verbose=0,
                                             save_best_only=True))

        logger = Log_callback(self.verbose)
        callbacks.append(logger)

        self.model.fit(X, y, batch_size=self.batch_size,
                       nb_epoch=self.max_epoch, callbacks=callbacks,
                       verbose=0, validation_data=eval_set[-1])

        self.best_iteration = logger.epoch - self.patience
        if self.save_path is not None:
            self.model.load_weights(self.save_path)

    def predic(self, X, **kwargs):
        return self.model.predict(X, self.batch_size, self.verbose)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X, self.batch_size, self.verbose)

