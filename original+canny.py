# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:27:01 2021

@author: Nafis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

train = pd.read_csv("training_set.csv")
test = pd.read_csv("test_set.csv")


# generating array of labels
labels = train["Label"].values
#encoding the categorical labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

#droping the labels from column and assigning the pixel value of images 
images = train.drop("Label",axis=1)
images = images.values
#converting each images pixel values in 28x28 array
images = np.array([np.reshape(i,(64,64)) for i in images])
images = np.array([i.flatten() for i in images])

images.shape

#slitting the dataset
x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size = .25, random_state = 18)

x_train.shape,x_test.shape

#normalizing pixel values
x_train = x_train / 255
x_test = x_test / 255

#adding bias neuron
x_train = x_train.reshape(x_train.shape[0], 64,64,1)
x_test = x_test.reshape(x_test.shape[0], 64,64,1)




from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


classifier = Sequential()
classifier.add(Conv2D(16, kernel_size=(3, 3), input_shape = (64,64,1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 38, activation = 'softmax'))

classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model = classifier.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=90,batch_size=64,validation_split=0.1)

classifier.summary()
classifier.save('BdSL_resized_original.h5')



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
y_pred = classifier.predict(x_test)
result = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test.argmax(axis=1), y_pred.round().argmax(axis=1))
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred.round())
print("Accuracy:",result2)

import matplotlib.pyplot as plt

plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model accuracy')
plt.show()

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model Loss')
plt.show()