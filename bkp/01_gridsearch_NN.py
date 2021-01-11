# -*- coding: utf-8 -*-

import os
import datetime
import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.grid_search import ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, **kwargs):
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
                       verbose=verbose, validation_data=eval_set[-1])

        self.best_iteration = logger.epoch - self.patience
        if self.save_path is not None:
            self.model.load_weights(self.save_path)

    def predic(self, X, **kwargs):
        return self.model.predict(X, self.batch_size, self.verbose)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X, self.batch_size, self.verbose)


def train_es(learner, X_train, y_train, X_test, y_test, esr=50):
    if type(learner) == BinaryMLP:
        learner.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=esr,
                    verbose=True)
        n_estimators = learner.best_iteration
        probs = learner.predict_proba(X_test, ntree_limit=n_estimators)[:, -1]
    else:
        learner.fit(X_train, y_train)
        if hasattr(learner, "predict_proba"):
            probs = learner.predict_proba(X_test)[:, -1]
        else:
            probs = learner.decision_function(X_test)
        n_estimators = 1
    return n_estimators, probs

################################# CONFIG ######################################

#base_path = "C:/Users/Rafael/Documents/data/DMC/data/"
base_path = "C:/Users/palbani/Desktop/Data Mining Cup/"
data_path = base_path + "raw_wvalid/"
target_path = base_path + "gridsearch/"

if not os.path.exists(target_path):
        os.mkdir(target_path)

index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]

drop_cols = index_cols + ["customerID", "paymentMethod",
                          "voucherID", "orderDate", "returnQuantity"]

learners = [
    {
        "learner": LogisticRegression,
        "params": {
            "penalty": ["l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 100],
            },
        "control_params": {
            "solver": "sag"
        },
        "fit_params": {}
    },
    {
        "learner": SGDClassifier,
        "params": {
            "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
            },
        "control_params": {},
        "fit_params": {}
    },
    {
        "learner": BinaryMLP,
        "params": {
            "hidden": [(50, 25), (30, 15), (20, 10), (10, 5)],
            "drop" : [(0, 0)],
            "activations": [("relu", "tanh"), ("tanh", "relu"),
                            ("relu", "sigmoid"), ("sigmoid", "relu"),
                            ("tanh", "sigmoid"), ("sigmoid", "tanh")]
            },
        "control_params": {
            "loss": "mean_absolute_error",
            "save_path": target_path + "NN.h",
            "max_epoch": 5000
        },
        "fit_params": {
            "esr": 10
        }
    },
]

################################# SCRIPT ######################################

file_name = os.path.basename(__file__).split(".")[0]

logging.basicConfig(filename=target_path + file_name + ".log",
                    level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("***** Iniciando rutina de busqueda de hiperparametros *****")

resultados_path = target_path + "resultado_NN.xlsx"
probs_path = target_path + "fold_3_NN.p"

try:
    path = "fold_3"
    logging.info("Leyendo datos...")
    X_train = pd.read_pickle(data_path + path + "/train.p")
    y_train = X_train.returnQuantity
    X_train = X_train.drop(drop_cols, axis=1).values.astype(np.float32, order="C")

    X_test = pd.read_pickle(data_path + path + "/test.p")
    y_test = X_test.returnQuantity

    if os.path.exists(resultados_path):
        resultados = pd.read_excel(resultados_path)
        resultados["params"] = [eval(d) for d in resultados.params]
        probs = pd.read_pickle(probs_path)
    else:
        resultados = pd.DataFrame([])
        probs = X_test[index_cols].copy()

    X_test = X_test.drop(drop_cols, axis=1).values.astype(np.float32, order="C")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    while True:
        is_new = False
        sorteos = 0
        while not is_new:
            if sorteos > 100:
                raise Exception("Se sortearon {0} algoritmos, sin encontra uno nuevo".format(sorteos))
            else:
                print("sorteando...")
            candidate = np.random.choice(learners)
            params = list(ParameterSampler(candidate["params"], 1))[0]
            learner_kind = candidate["learner"].__name__
            old = resultados.filter(regex=learner_kind, axis=0)
            learner_name = learner_kind + "_" + str(len(old) + 1)
            if (len(old) == 0) or not np.any(old.params == params):
                is_new = True
                sorteos = 0
            else:
                sorteos += 1

        used_params = params.copy()
        used_params.update(candidate["control_params"])
        logging.info("-"*60)
        start = datetime.datetime.now()
        logging.info("Entrenando learner: " + learner_name)
        logging.info("parametros:" + str(used_params))
        learner = candidate["learner"](**used_params)

        n_estimators, probs[learner_name] = train_es(learner,
                                                     X_train,
                                                     (y_train > 0).astype(int),
                                                     X_test,
                                                     (y_test > 0).astype(int),
                                                     **candidate["fit_params"])

        if hasattr(learner, "predict_proba"):
            resultado = mean_absolute_error(y_test.values,
                                            probs[learner_name].values >= 0.5)
        else:
            resultado = mean_absolute_error(y_test.values,
                                            probs[learner_name].values >= 0)
        logging.info("Resultado en {0} fue de {1}".format(path, resultado))

        enlapsed = (datetime.datetime.now() - start).seconds / 60.0
        enlapsed = round(enlapsed, 2)
        logging.info("Entrenado en {0} minutos. Guardando resultados...".format(enlapsed))
        cols = ["train_minutes", "resultado", "params", "used_params"]
        vals = [enlapsed, resultado, params, used_params]
        for p in params:
            cols.append("param_" + p)
            vals.append(params[p])
        if "n_estimators" in used_params:
            used_params["n_estimators"] = n_estimators
            cols.append("param_n_estimators")
            vals.append(n_estimators)
        if "max_epoch" in used_params:
            del used_params["save_path"]
            used_params["early_stop"] = False
            used_params["max_epoch"] = n_estimators
            cols.append("param_max_epoch")
            vals.append(n_estimators)
        vals = np.asarray(vals)
        vals = vals.reshape((1, vals.shape[0]))
        resultados = resultados.append(pd.DataFrame(vals, columns=cols,
                                                    index=[learner_name]))
        probs.to_pickle(probs_path)
        resultados.to_excel(resultados_path)

except Exception as e:
    logging.exception("***** Se ha encontrado un error *****")
finally:
    import winsound
    winsound.Beep(1000, 600)
    logging.info("***** TERMINADO *****")
