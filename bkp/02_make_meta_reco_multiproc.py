# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:37:29 2016

@author: Rafael
"""

import os
import datetime

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

#base_path = "C:/Users/Rafael/Documents/data/DMC/data/"
base_path = "C:/Users/rcrescenzi/Documents/Personal/data/DMC/data/"
data_path = base_path + "raw/"
target_path = base_path + "meta_reco/"

def t_func(i, fold):
    if os.path.exists(target_path + "fold_" + str(i + 1) + ".p"):
        return 0

    f_path = target_path + "fold_" + str(i+1) + ".log"
    if os.path.exists(f_path):
        os.remove(f_path)

    logging.basicConfig(filename=f_path,
                        level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    index_cols = ["orderID", "articleID", "colorCode", "sizeCode", "customerID"]

    if i < 3:
        train = pd.read_csv(data_path + "train.txt", sep=";", na_values="NA",
                            keep_default_na=False, parse_dates=[1])
        valid = train[(train.orderDate >= fold[1][0]) & (train.orderDate <= fold[1][1])]
        train = train[train.orderDate <= fold[0]]
    else:
        train = pd.read_csv(data_path + "train.txt", sep=";", na_values="NA",
                            keep_default_na=False, parse_dates=[1])
        valid = pd.read_csv(data_path + "test.txt", sep=";", na_values="NA",
                            keep_default_na=False, parse_dates=[1])

    train_q = train[["customerID", "articleID", "quantity", "returnQuantity"]].copy()
    train_q.set_index("customerID", inplace=True)
    orders = valid.orderID.unique().shape[0]

    valid_res = valid[index_cols].copy()
    valid_res["ret_ratio"] = 0
    valid_res["jac_ret"] = 0
    valid_res["jac_noret"] = 0
    valid_res["rebuy"] = 0
    valid_res["returned"] = 0
    valid_res.set_index(["orderID", "articleID"], inplace=True)

    baskets = train.groupby("customerID").articleID.apply(np.unique).map(set)
    valid_baskets = valid.groupby("customerID").articleID.apply(np.unique).map(set)
    p_order = 0
    start = datetime.datetime.now()
    del train, valid
    for orderID, case in valid_res.groupby(level="orderID"):
        p_order += 1
        customerID = case.customerID.unique()[0]
        custumer_baskets = valid_baskets.ix[customerID]
        if customerID in baskets.index:
            custumer_baskets = custumer_baskets.union(baskets.ix[customerID])
        for articleID in case.reset_index().articleID.unique():
            candidates = baskets.ix[train_q[train_q.articleID == articleID].index.unique()]
            if len(candidates) == 0: continue
            returned = train_q.ix[candidates.index]
            returned = returned[returned.articleID == articleID]
            returned = returned.reset_index().groupby("customerID").sum()
            returned = (returned.returnQuantity / returned.quantity).fillna(0)
            returned.name = "returned"

            jac_dist = candidates.map(lambda x: len(custumer_baskets.intersection(x)) / len(custumer_baskets.union(x)))
            jac_dist.name = "jac_dist"

            valid_res.ix[(orderID, articleID), "ret_ratio"] = pd.concat([jac_dist, returned], axis=1).prod(axis=1).mean()
            temp = pd.concat([jac_dist, returned>=0.5], axis=1).groupby("returned").mean()
            if True in temp.index:
                valid_res.ix[(orderID, articleID), "jac_ret"] = temp.ix[True].values[0]
            if False in temp.index:
                valid_res.ix[(orderID, articleID), "jac_noret"] = temp.ix[False].values[0]

            if customerID in candidates.index:
                valid_res.ix[(orderID, articleID), "rebuy"] = True
                returned = train_q.ix[[customerID]]
                returned = returned[returned.articleID == articleID]
                if returned.quantity.sum() > 0:
                    returned = returned.returnQuantity.sum() / returned.quantity.sum()
                else:
                    returned = min(1, returned.returnQuantity.sum())
                valid_res.ix[(orderID, articleID), "returned"] = returned
        if p_order % 100 == 0:
            enlapsed = (datetime.datetime.now() - start).seconds / 60.0
            pct = round(p_order / orders * 100, 2)
            rm = round((enlapsed / p_order * orders - enlapsed) / 60, 2)
            logging.info("hecho {0}% del fold {1}, tiempo restante: {2} horas".format(pct, i + 1, rm))

    valid_res["jac_ratio"] = valid_res.jac_ret / (valid_res.jac_noret + 1e-35)
    valid_res.drop("customerID", axis=1, inplace=True)
    if i < 3:
        valid_res.reset_index().to_pickle(target_path + "fold_" + str(i + 1) + ".p")
    else:
        valid_res.reset_index().to_pickle(target_path + "test.p")
    logging.info("terminado fold {0}".format(i + 1))
    return valid_res

folds = [["2014-12-31", ("2015-1-01", "2015-3-31")],
         ["2015-3-31", ("2015-4-01", "2015-6-30")],
         ["2015-6-30", ("2015-7-01", "2015-9-30")],
         ["2015-9-30", ("2015-10-01", "2015-12-31")]]

if __name__ == "__main__":
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    pool = Parallel(n_jobs=4)
    a = pool(delayed(t_func)(i, fold) for i, fold in enumerate(folds))
