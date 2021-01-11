# -*- coding: utf-8 -*-

import os
import datetime
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.grid_search import ParameterSampler
from sklearn.metrics import mean_absolute_error

def train_es(learner, X_train, y_train, X_test, y_test, esr=50):
    max_n_estimators = learner.n_estimators
    learner.set_params(warm_start = True)
    learner.set_params(n_estimators = 0)
    old_res = 1
    new_res = 0.99999999
    while (new_res < old_res) and \
          (learner.n_estimators + esr <= max_n_estimators):
        learner.set_params(n_estimators = learner.n_estimators + esr)
        old_res = new_res
        learner.fit(X_train, y_train)
        probs = learner.predict_proba(X_test)[:, -1]
        new_res = mean_absolute_error(y_test.values, probs >= 0.5)
        print("resultado con {0} estimadores: {1}".format(learner.n_estimators,
                                                          new_res))
    n_estimators = learner.n_estimators
    if (new_res > old_res):
        n_estimators -= esr
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
        "learner": RandomForestRegressor,
        "params": {
            "max_depth": [3, 4, 5, 10, 15],
            "max_features": ["auto", 0.25, 0.5, 0.75, 1],
            "min_weight_fraction_leaf": [0, 0.0001, 0.001, 0.01],
            "bootstrap": [True, False]
            },
        "control_params": {
            "n_estimators": 1000,
            "n_jobs": -1
        },
        "fit_params": {
            "esr": 50
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

resultados_path = target_path + "resultados_RF.xlsx"
probs_path = target_path + "fold_3_RF.p"

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
