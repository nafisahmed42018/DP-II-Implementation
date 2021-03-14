# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 07:47:38 2021

@author: Nafis
"""
#except j and z
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

# generating array of labels
labels = train["label"].values
#encoding the categorical labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

#droping the labels from column and assigning the pixel value of images 
images = train.drop("label",axis=1)
images = images.values
#converting each images pixel values in 28x28 array
images = np.array([np.reshape(i,(28,28)) for i in images])
images = np.array([i.flatten() for i in images])

images.shape

#slitting the dataset
x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size = .25, random_state = 18)

x_train.shape,x_test.shape

#normalizing pixel values
x_train = x_train / 255
x_test = x_test / 255



knn_classifier = KNeighborsClassifier()
model = knn_classifier.fit(x_train,y_train)
y_pred = knn_classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
result = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
