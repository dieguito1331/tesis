# -*- coding: utf-8 -*-

import os
import datetime
import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error

################################# CONFIG ######################################

#base_path = "C:/Users/Rafael/Documents/data/DMC/data/"
base_path = "C:/Users/rcrescenzi/Documents/Personal/data/DMC/data/"
data_path = base_path + "raw_wvalid/"
target_path = base_path + "gridsearch/"

if not os.path.exists(target_path):
        os.mkdir(target_path)

index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]
drop_cols = index_cols + ["returnQuantity"]

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
    resultados = pd.read_excel(resultados_path)
    resultados["params"] = [eval(d) for d in resultados.params]

    y_train = pd.read_pickle(data_path + path + "/test.p")[drop_cols]
    X_train =  pd.merge(pd.read_pickle(probs_path), y_train, on=index_cols)

    res = pd.DataFrame([], index=resultados.index)

    learner = DecisionTreeClassifier()
    learner.fit(X_train.drop(drop_cols, axis=1), X_train.returnQuantity > 0)
    res["fi"] = learner.feature_importances_

    learner = LogisticRegression()
    y_train = X_train.returnQuantity
    X_train = X_train.drop(drop_cols, axis=1).clip(1e-5, 1 - 1e-5)
    X_train = np.log(X_train / (1 - X_train))
    learner.fit(X_train, y_train > 0)
    res["coef"] = learner.coef_.T

except Exception as e:
    logging.exception("***** Se ha encontrado un error *****")
finally:
    import winsound
    winsound.Beep(1000, 600)
    logging.info("***** TERMINADO *****")
