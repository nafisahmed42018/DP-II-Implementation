# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 07:26:56 2021

@author: Nafis
"""

import pandas as pd


train = pd.read_csv("0.csv")


for i in range(1,38):
    df = pd.read_csv(f"{i}.csv")
    train = train.append(df ,ignore_index = True)


train.to_csv("training_set.csv",index=False)


