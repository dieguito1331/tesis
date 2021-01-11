# -*- coding: utf-8 -*-

import os
import datetime
import logging

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from Perceptron import BinaryMLP

from sklearn.grid_search import ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from utils import train_es

################################# CONFIG ######################################

base_path = "C:/Users/Rafael/Documents/data/DMC/data/"
#base_path = "C:/Users/rcrescenzi/Documents/Personal/data/DMC/data/"
data_path = base_path + "raw_wvalid/"
train_dirs = os.listdir(data_path)
target_path = base_path + "gridsearch/"

if not os.path.exists(target_path):
        os.mkdir(target_path)

index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]

drop_cols = index_cols + ["customerID", "paymentMethod",
                          "voucherID", "orderDate", "returnQuantity"]

learners = [
    {
        "learner": XGBClassifier,
        "params": {
            "learning_rate": [0.1],
            "colsample_bytree": [0.15, 0.25, 0.5, 0.75, 1],
            "colsample_bylevel": [1],
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25],
            "gamma": [0, 0.01, 0.05, 0.1, 0.25, 0.5],
            "subsample": [0.5, 0.75, 1],
            "min_child_weight": [1, 5, 10, 15, 25, 50, 100],
            "base_score": [0.52]
            },
        "control_params": {
            "n_estimators": 1000
        }
    },
#    {
#        "learner": RandomForestClassifier,
#        "params": {
#            "max_depth": [3, 4, 5, 10, 25, None],
#            "max_features": ["auto", 0.1, 0.25, 0.5, 0.75, 1],
#            "min_weight_fraction_leaf": [0, 0.01, 0.001],
#            "bootstrap": [True, False]
#            },
#        "control_params": {
#            "n_estimators": 1000,
#            "n_jobs": -1
#        }
#    },
#    {
#        "learner": ExtraTreesClassifier,
#        "params": {
#            "max_depth": [None],
#            "max_features": ["auto", 0.1],
#            "min_weight_fraction_leaf": [0, 0.01, 0.001],
#            "bootstrap": [True, False]
#            },
#        "control_params": {
#            "n_estimators": 1000,
#            "n_jobs": -1
#        }
#    },
#    {
#        "learner": LogisticRegression,
#        "params": {
#            "penalty": ["l2"],
#            "C": [0.001, 0.01, 0.1, 1, 10, 100, 100],
#            },
#        "control_params": {
#            "solver": "sag"
#        }
#    },
#    {
#        "learner": SGDClassifier,
#        "params": {
#            "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
#            },
#        "control_params": {}
#    },
#    {
#        "learner": BinaryMLP,
#        "params": {
#            "hidden": [(50, 25), (30, 15), (20, 10)],
#            "drop" : [(0, 0)],
#            "activations": [("relu", "tanh"), ("tanh", "relu"),
#                            ("tanh", "sigmoid"), ("sigmoid", "tanh")]
#            },
#        "control_params": {
#            "loss": "mean_absolute_error",
#            "save_path": target_path + "NN.h"
#        }
#    },
]

################################# SCRIPT ######################################

file_name = os.path.basename(__file__).split(".")[0]

logging.basicConfig(filename=target_path + file_name + ".log",
                    level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("***** Iniciando rutina de busqueda de hiperparametros *****")

resultados_path = target_path + "resultados.xlsx"
probs_path = target_path + "gs_probs.p"

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
    scaler.fit(np.r_[X_train, X_test])
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
                                                     (y_test > 0).astype(int))

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
            cols.append("param_n_estimators" + p)
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
