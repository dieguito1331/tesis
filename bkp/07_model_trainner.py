# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from Perceptron import BinaryMLP

from sklearn.preprocessing import StandardScaler

################################# CONFIG ######################################

#base_path = "C:/Users/Rafael/Documents/data/DMC/data/"
base_path = "C:/Users/palbani/Desktop/Data Mining Cup/"

excel_file = "resultados_ET.xlsx"

data_path = base_path + "raw_wvalid/"
target_path = base_path + "probs_stage_1/"

train_dirs = os.listdir(data_path)

if not os.path.exists(target_path):
        os.mkdir(target_path)

index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]

drop_cols = index_cols + ["customerID", "paymentMethod",
                          "voucherID", "orderDate", "returnQuantity"]

################################# SCRIPT ######################################

file_name = os.path.basename(__file__).split(".")[0]

logging.basicConfig(filename=target_path + file_name + ".log",
                    level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("***** Iniciando rutina de entrenamiento *****")

resultados = pd.read_excel(target_path + excel_file)
resultados["used_params"] = [eval(d) for d in resultados.used_params]
escalar = resultados.escalar[0]

try:
    for path in train_dirs:
        probs_path = target_path + path + ".p"

        logging.info("-"*60)
        logging.info("Leyendo datos para {0}...".format(path))
        X_train = pd.read_pickle(data_path + path + "/train.p")
        y_train = (X_train.returnQuantity > 0).astype(int)
        X_train = X_train.drop(drop_cols, axis=1).values.astype(np.float32, order="C")

        X_test = pd.read_pickle(data_path + path + "/test.p")

        if os.path.exists(probs_path):
            probs = pd.read_pickle(probs_path)
        else:
            probs = X_test[index_cols].copy()

        X_test = X_test.drop(drop_cols, axis=1).values.astype(np.float32, order="C")

        if escalar:
            logging.info("-"*60)
            logging.info("Scaling...")
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        for learner_name, case in resultados.iterrows():

            if learner_name in probs.columns: continue

            logging.info("-"*60)
            used_params = case["used_params"]
            logging.info("Entrenando learner: " + learner_name)
            logging.info("parametros:" + str(used_params))
            logging.info("tiempo esperado: " + str(case.train_minutes))

            learner = eval(learner_name.split("_")[0])(**used_params)
            learner.fit(X_train, y_train)

            if hasattr(learner, "predict_proba"):
                probs[learner_name] = learner.predict_proba(X_test)[:, -1]
            else:
                probs[learner_name] = learner.decision_function(X_test)

            logging.info("Guardando resultados...")
            probs.to_pickle(probs_path)

except Exception as e:
    logging.exception("***** Se ha encontrado un error *****")
finally:
    import winsound
    winsound.Beep(1000, 600)
    logging.info("***** TERMINADO *****")

