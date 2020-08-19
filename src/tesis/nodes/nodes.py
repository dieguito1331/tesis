import pandas as pd
from collections import defaultdict
import numpy as np
import logging
import os
from functools import reduce
import datetime

def reduce_join(df, columns):
    assert len(columns) > 1
    slist = [df[x].astype(str) for x in columns]
    return reduce(lambda x, y: x + '' + y, slist[1:], slist[0])

def encode(train, c, t):
    default_ratio = (train[t] > 0).mean()
    general_var = (train[t] > 0).var()
    ct = pd.crosstab(train[c], train[t] > 0)
    ct.columns = [0, 1]
    ct["n"] = ct.sum(axis=1)
    ct["p"] = ct[1] / ct.n
    temp = train.groupby(c)[t].var().fillna(0)
    temp.name = "local_var"
    ct = pd.concat([ct, temp], axis=1)
    ct["pond"] = (ct.n * ct.local_var) / (general_var + ct.n * ct.local_var)
    ct["final"] = ct.pond * ct.p + (1 - ct.pond) * default_ratio
    return defaultdict(lambda: default_ratio, ct.final.to_dict())

def createSimpleVariables(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Comienzo con la creacion de las variables simples")
    temp = data.groupby("orderID").price.sum()
    temp.name = "orderPrice"
    data["orderPrice"] = data.join(temp, on= "orderID")["orderPrice"]
    temp = data.groupby("orderID").quantity.sum()
    temp.name = "orderQuantity"
    data["orderQuantity"] = data.join(temp, on="orderID")["orderQuantity"]
    data["unitPrice"] = (data.price / data.quantity).fillna(0)
    data["rrp"].fillna(data.unitPrice, inplace=True)
    data["productGroup"].fillna(0, inplace=True)
    data["voucherID"].fillna("0", inplace=True)
    data["rrpPriceRatio"] = (data.unitPrice / data.rrp).fillna(0)
    data["orderUnitPriceRatio"] = (data.unitPrice / data.orderPrice).fillna(0)
    data["orderPriceRatio"] = (data.price / data.orderPrice).fillna(0)
    data["orderQuantityRatio"] = (data.quantity / data.orderQuantity).fillna(0)
    data["orderDiscountPercent"] = (data.voucherAmount / data.orderPrice).fillna(0)
    data["paidPrice"] = data.price * (1 - data.orderDiscountPercent)
    temp = data.groupby("articleID").unitPrice.mean()
    temp.name = "artMeanPrice"
    data["mean_price_ratio"] = (data.unitPrice / data.join(temp, on="articleID")["artMeanPrice"]).fillna(0)
    data["orderDate"] = pd.to_datetime(data.orderDate)
    data["month"]  = data.orderDate.dt.month
    data["week"] = data.orderDate.dt.week
    data["day"] = data.orderDate.dt.day
    data["fortnight"] = (data.month - 1) * 2 + (data.day > 15)
    data["weekday"] = data.orderDate.dt.weekday
    temp = data.groupby(["orderID", "articleID"])
    temp_data = temp.quantity.sum()
    temp_data.name = "itemQuantity"
    data = data.join(temp_data, on=["orderID", "articleID"])
    data["itemQuantityRatio"] = (data.quantity / data.itemQuantity).fillna(0)
    temp = data.groupby(["orderID", "articleID", "colorCode"])
    temp_data = temp.quantity.sum()
    temp_data.name = "itemQuantityColor"
    data = data.join(temp_data, on=["orderID", "articleID", "colorCode"])
    temp = data.groupby(["orderID", "articleID", "sizeCode"])
    temp_data = temp.quantity.sum()
    temp_data.name = "itemQuantitySize"
    data = data.join(temp_data, on=["orderID", "articleID", "sizeCode"])
    data["itemDifSize"] = data.itemQuantity != data.itemQuantitySize
    data["itemDifColor"] = data.itemQuantity != data.itemQuantityColor
    for c in data.columns:
        if data[c].dtype not in ["object", "datetime64[ns]"]:
            data[c] = data[c].astype(np.float32)
    logging.info("Finalizo la creacion de las variables simples")
    return data


def createBayesianSingleVariables(train: pd.DataFrame,test: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    logging.info("Comienzo con la creacion de las variables usando Metodo bayesiano (Simples)")

    for c in parameters["codify_cols"]:
        ct = encode(train[[c, "returnQuantity"]], c, "returnQuantity")
        train[c + "_encoded"] =  train[c].apply(lambda x: ct[x]).astype(np.float32)
        test[c + "_encoded"] =  test[c].apply(lambda x: ct[x]).astype(np.float32)
    train.to_csv("data/04_features/dataBase.txt", sep=";", index=False)
    logging.info("Finalizo la creacion de las variables usando Metodo bayesiano (Simples)")
    return train, test
    

def createBayesianMultipleVariables(train: pd.DataFrame,test: pd.DataFrame, parameters: dict) -> bool:
    for c in parameters["codify_cols_group"]:
        variableName = "-".join(c)
        train = train.join(reduce_join(train, c).to_frame()).rename(columns={0:variableName})
        test = test.join(reduce_join(test, c).to_frame()).rename(columns={0:variableName})
        ct = encode(train[[variableName, "returnQuantity"]], variableName, "returnQuantity")
        train[variableName + "_encoded"] =  train[variableName].apply(lambda x: ct[x]).astype(np.float32)
        test[variableName + "_encoded"] =  test[variableName].apply(lambda x: ct[x]).astype(np.float32)
        train = train.drop(columns=[variableName])
        test = test.drop(columns=[variableName])
    train.to_csv("data/04_features/trainDataProcessing.txt.gz", sep=";", index=False, compression = "gzip")
    test.to_csv("data/04_features/testDataProcessing.txt.gz", sep=";", index=False, compression = "gzip")

    return train, test

def createFolds(train: pd.DataFrame, test: pd.DataFrame, parameters: dict) -> bool:
    data = pd.concat([train, test])
    data.index = range(data.shape[0])
    folds = parameters["folds"]
    for i, fold in enumerate(folds):
        path = "data/05_model_input/fold_{}".format(str(i+1))
        if not os.path.exists(path):
            os.mkdir(path)
        train[data.orderDate <= fold[0]].to_pickle(path + "/train.p")
        data[(data.orderDate >= fold[1][0]) & (data.orderDate <= fold[1][1])].to_pickle(path + "/test.p")

    return True

def createMetaReco(parameters: dict) -> bool:
    folds = [y[1] for y in [x[0].split("\\") for x in os.walk("data/05_model_input")][1:]]
    index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]
    for i, fold in enumerate(folds):
        valid = pd.read_pickle("data/05_model_input/{}/test.p".format(fold))
        train = pd.read_pickle("data/05_model_input/{}/train.p".format(fold))
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
        for orderID, case in valid.groupby("orderID"):
            p_order += 1
            customerID = case.customerID.unique()[0]
            custumer_baskets = valid_baskets.ix[customerID]
            if customerID in baskets.index:
                custumer_baskets = custumer_baskets.union(baskets.ix[customerID])
            for articleID in case.articleID.unique():
                candidates = baskets.ix[train[train.articleID == articleID].customerID.unique()]
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

        path = "data/07_model_output/metaReco/{}".format(fold)
        if not os.path.exists(path):
            os.mkdir(path)

        valid_res["jac_ratio"] = valid_res.jac_ret / (valid_res.jac_noret + 1e-35)    
        valid_res.reset_index().to_pickle(path+"test.p")
    return True
 
#def trainXGBoostModels(parameters:dict):