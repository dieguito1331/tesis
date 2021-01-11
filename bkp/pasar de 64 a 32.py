# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:09:25 2016

@author: palbani
"""

import numpy as np
import pandas as pd

index_cols = ["orderID", "articleID", "colorCode", "sizeCode"]
pd.read_pickle("C:/Users/palbani/Desktop/Data Mining Cup/gridsearch/fold_3_NN_64.p").set_index(index_cols).astype(np.float32).reset_index().to_pickle("C:/Users/palbani/Desktop/Data Mining Cup/gridsearch/fold_3_NN.p")

pd.read_pickle("C:/Users/palbani/Desktop/Data Mining Cup/gridsearch/fold_3_servidor_64.p").set_index(index_cols).astype(np.float32).reset_index().to_pickle("C:/Users/palbani/Desktop/Data Mining Cup/gridsearch/fold_3_servidor.p")

