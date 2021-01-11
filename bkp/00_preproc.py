# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from utils import encode

base_path = "C:/Users/Rafael/Documents/data/DMC/data/"
data_path = base_path + "raw/"
target_path = base_path + "raw_wvalid/"

train = pd.read_csv(data_path + "train.txt", sep=";", na_values="NA",
                    keep_default_na=False, parse_dates=[1])
test = pd.read_csv(data_path + "test.txt", sep=";", na_values="NA",
                    keep_default_na=False, parse_dates=[1])

data = pd.concat([train, test])
data.index = range(data.shape[0])
del train, test

codify_cols = ["articleID", "colorCode", "sizeCode",
               "customerID", "paymentMethod", "voucherID",
               "deviceID", "productGroup"]

temp = data.groupby("orderID").price.sum()
temp.name = "order_value"
data["order_value"] = data.join(temp, on="orderID")["order_value"]

temp = data.groupby("orderID").quantity.sum()
temp.name = "order_quantity"
data["order_quantity"] = data.join(temp, on="orderID")["order_quantity"]

data["unit_price"] = (data.price / data.quantity).fillna(0)
data["rrp"].fillna(data.unit_price, inplace=True)
data["productGroup"].fillna(0, inplace=True)
data["voucherID"].fillna("0", inplace=True)
data["rrp_price_ratio"] = (data.unit_price / data.rrp).fillna(0)
data["order_unit_price_ratio"] = (data.unit_price / data.order_value).fillna(0)
data["order_price_ratio"] = (data.price / data.order_value).fillna(0)
data["order_quant_ratio"] = (data.quantity / data.order_quantity).fillna(0)

data["order_discount_percent"] = (data.voucherAmount / data.order_value).fillna(0)
data["paid_price"] = data.price * (1 - data.order_discount_percent)

temp = data.groupby("articleID").unit_price.mean()
temp.name = "art_mean_price"
data["mean_price_ratio"] = (data.unit_price / data.join(temp, on="articleID")["art_mean_price"]).fillna(0)

data["month"]  = data.orderDate.dt.month
data["week"] = data.orderDate.dt.week
data["day"] = data.orderDate.dt.day
data["quincena"] = (data.month - 1) * 2 + (data.day > 15)
data["weekday"] = data.orderDate.dt.weekday

temp = data.groupby(["orderID", "articleID"])
temp_data = temp.quantity.sum()
temp_data.name = "item_quantity"
data = data.join(temp_data, on=["orderID", "articleID"])
data["item_quantity_ratio"] = (data.quantity / data.item_quantity).fillna(0)

temp = data.groupby(["orderID", "articleID", "colorCode"])
temp_data = temp.quantity.sum()
temp_data.name = "item_quantity_color"
data = data.join(temp_data, on=["orderID", "articleID", "colorCode"])

temp = data.groupby(["orderID", "articleID", "sizeCode"])
temp_data = temp.quantity.sum()
temp_data.name = "item_quantity_size"
data = data.join(temp_data, on=["orderID", "articleID", "sizeCode"])

data["item_dif_size"] = data.item_quantity != data.item_quantity_size
data["item_dif_color"] = data.item_quantity != data.item_quantity_color

for c in data.columns:
    if data[c].dtype not in ["object", "datetime64[ns]"]:
        data[c] = data[c].astype(np.float32)

folds = [["2014-12-31", ("2015-1-01", "2015-3-31")],
         ["2015-3-31", ("2015-4-01", "2015-6-30")],
         ["2015-6-30", ("2015-7-01", "2015-9-30")],
         ["2015-9-30", ("2015-10-01", "2015-12-31")]]


for i, fold in enumerate(folds):
    for c in codify_cols:
        ct = encode(data[data.orderDate <= fold[0]][[c, "returnQuantity"]], c, "returnQuantity")
        data[c + "_encoded"] =  data[c].apply(lambda x: ct[x]).astype(np.float32)
    tdir = target_path + "fold_" + str(i + 1) + "/"
    if not os.path.exists(tdir):
        os.mkdir(tdir)
    data[data.orderDate <= fold[0]].to_pickle(tdir + "train.p")
    data[(data.orderDate >= fold[1][0]) & (data.orderDate <= fold[1][1])].to_pickle(tdir + "test.p")
