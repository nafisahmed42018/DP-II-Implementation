# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 03:05:38 2021

@author: Nafis
"""


import numpy as np
import cv2
import os
import pandas as pd

for i in range(0,38):
    
    instances = []
    for filepath in os.listdir(f'./RESIZED_DATASET/{i}/'):
        img = cv2.imread(f'./RESIZED_DATASET/{i}/{filepath}',0)
        resize = cv2.resize(img,(64,64))
        blur = cv2.bilateralFilter(resize,4,75,75)
        opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, (5,5))
        canny = cv2.Canny(opening, 75, 75)
        instances.append(canny)
    
    instances = np.reshape(instances, (len(instances),4096))
    df = pd.DataFrame(instances)
    df["Label"] = i
    df.to_csv(f"{i}.csv",index=False)



'''
instances = []
for filepath in os.listdir('./RESIZED_TESTING_DATA/38/'):
    img = cv2.imread('./RESIZED_TESTING_DATA/38/{0}'.format(filepath),1)
    blur = cv2.bilateralFilter(img,4,75,75)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, (5,5))
    canny = cv2.Canny(opening, 75, 75)
    instances.append(canny)

instances = np.reshape(instances, (len(instances),50176))
thirtyeight = pd.DataFrame(instances)
thirtyeight["Label"] = 38
thirtyeight.to_csv("thirtyeight.csv",index=False)

'''

