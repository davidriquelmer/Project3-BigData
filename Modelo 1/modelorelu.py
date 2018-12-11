from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


#reemplazar el nombre del csv
data = pd.read_csv("cleantextlabels7.csv")
testData = pd.read_csv("dataproyecto2.csv", dtype=object, error_bad_lines=False)
data.head()
data['tags'].value_counts()

#porcentaje de clasificacion 0.8 a 0.2
train_size = int(len(data) * .8)

train_posts = data['post'][:train_size]
train_tags = data['tags'][:train_size]
test_posts = data['post'][train_size:]
test_tags = data['tags'][train_size:]
sizepy2 = len(testData)
dataproyecto2 = testData['words'][:sizepy2]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) 
x_proyecto2 = tokenize.texts_to_matrix(dataproyecto2)
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
batch_size = 32
epochs = 4


#Construyendo nuestro modelo
model = Sequential()
model.add(Dense(256, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dense(256, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Evaluar modelo de precision
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

text_labels = encoder.classes_ 

salida = []

for i in range(sizepy2):
    prediction = model.predict(np.array([x_proyecto2[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
	post = project_posts.iloc[i]
    row = (str(post), str(predicted_label))
    postsArr.append(row)
    print(test_posts.iloc[i][:50], "...")
    print('Actual label:' + test_tags.iloc[i])
    print("Predicted label: " + predicted_label + "\n")
	
df = pd.DataFrame(np.array(salida))
df.columns = ['Post', 'Tag']
df.to_csv("tag.csv")

df = spark.read.format("com.databricks.spark.csv").option("header", "true").load("tag.csv")
df.createOrReplaceTempView("red")
labels = spark.sql("select label, count(*) from red group by label order by label desc")
labels.show()
labels.write.csv('conteo1.csv')
	
