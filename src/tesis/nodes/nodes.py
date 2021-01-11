import pandas as pd
from collections import defaultdict
import numpy as np
import logging
import os
from functools import reduce
import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import ParameterSampler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


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
 

def train_es(learner, X_train, y_train, X_test, y_test, esr=20):
    if type(learner) in [XGBClassifier, BinaryMLP]:
        learner.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=esr,
                    verbose=True)
        n_estimators = learner.best_iteration
        probs = learner.predict_proba(X_test, ntree_limit=n_estimators)[:, -1]
    elif type(learner) in [RandomForestClassifier, ExtraTreesClassifier,
                           AdaBoostClassifier, BaggingClassifier]:
        max_n_estimators = learner.n_estimators
        learner.set_params(warm_start = True)
        learner.set_params(n_estimators = 50)
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
    else:
        learner.fit(X_train, y_train)
        if hasattr(learner, "predict_proba"):
            probs = learner.predict_proba(X_test)[:, -1]
        else:
            probs = learner.decision_function(X_test)
        n_estimators = 1
    return n_estimators, probs


def trainModels(parameters: dict):
    index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]
    drop_cols = index_cols + ["customerID", "paymentMethod",
                          "voucherID", "orderDate", "returnQuantity"]
    learners = parameters["learners"]
    path = "data/05_model_input/fold_3"
    results_path = "data/08_reporting"

    X_train = pd.read_pickle(path + "/train.p")
    X_train = X_train.drop(drop_cols, axis=1).values.astype(np.float32, order="C")
    y_train = X_train.returnQuantity
    
    X_test = pd.read_pickle(path + "/test.p")
    X_test = X_test.drop(drop_cols, axis=1).values.astype(np.float32, order="C")
    y_test = X_test.returnQuantity

    if os.path.exists(results_path+"resultados.xlsx"):
        results = pd.read_excel(results_path+"resultados.xlsx")
        results["params"] = [eval(d) for d in results.params]
        probs = pd.read_pickle(results_path+"gs_probs.p")
    else:
        resultados = pd.DataFrame([])
        probs = X_test[index_cols].copy()


    scaler = StandardScaler()
    scaler.fit(np.r_[X_train, X_test])
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    seguir = True
    modelsQuantity = 0
    while seguir:
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
        probs.to_pickle(results_path+"gs_probs.p")
        resultados.to_excel(results_path+"resultados.xlsx")
        modelsQuantity +=1
        if modelsQuantity >= 500:
            seguir = False

def modelSelection(parameters:dict):
    results_path = "data/08_reporting"
    path = "data/05_model_input/fold_3"
    index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]
    drop_cols = index_cols + ["returnQuantity"]

    resultados = pd.read_excel(results_path+"resultados.xlsx")
    resultados["params"] = [eval(d) for d in resultados.params]

    y_train = pd.read_pickle(path + "/test.p")[drop_cols]
    X_train =  pd.merge(pd.read_pickle(results_path+"gs_probs.p"), y_train, on=index_cols)

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

