# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 04:41:14 2021

@author: Nafis
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

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

#adding bias neuron
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)




from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(3, 3), input_shape = (28,28,1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(16, kernel_size=(3, 3), activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size = (4, 4)))



classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 24, activation = 'softmax'))

classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model = classifier.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=90,batch_size=64,validation_split=0.1)

classifier.summary()
classifier.save('mnist_cnn.h5')




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

"""
from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, x_test, y_test,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
"""