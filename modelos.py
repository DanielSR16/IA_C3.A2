from cProfile import label
import math
from pickletools import optimize
from typing import Counter
from unittest.mock import patch
import matplotlib.image as img
import os
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import cv2 

train_dir = 'frutas_verduras'
classes = os.listdir(train_dir)

x = []
y = []
name_clas = []
y_dic = {}
contador = 0
for class_name in classes : 
    print(contador)
    class_samples = os.listdir(f'{train_dir}/{class_name}')
    for sample in class_samples:

        image = f'{train_dir}/{class_name}/{sample}'

        image = cv2.resize(cv2.imread(f'{train_dir}/{class_name}/{sample}'),(30,30))
        imagen_normalizar = tf.cast(image,tf.float32)
        imagen_normalizar /= 255

        x.append(imagen_normalizar)
        y.append(contador)
        
    contador = contador +1 
    name_clas.append(class_name)
x = np.asarray(x)
y = np.asarray(y)

# print(x.shape , y.shape)
# print(name_clas)

def modelo_convolucional():
    print('Ejecutando modelo convolucional')
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(50,(3,3),input_shape= (30,30,3),activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(60,(3,3),activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(60,(3,3),activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(50,activation = tf.nn.relu),
        
        tf.keras.layers.Dense(116,activation = tf.nn.softmax),

    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    lotes = 32
    # x = x.repeat().batch(lotes)

    historial = model.fit(x,y, epochs=20, steps_per_epoch= math.ceil(len(x)/lotes))
 
    return historial.history['loss']



def modelo_denso_1():
    print('Ejecutando modelo Denso 1')
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape= (30,30,3)),
        tf.keras.layers.Dense(100,activation = tf.nn.relu),
        tf.keras.layers.Dense(100,activation = tf.nn.relu),
        tf.keras.layers.Dense(100,activation = tf.nn.relu),
        tf.keras.layers.Dense(116,activation = tf.nn.softmax),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    lotes = 32
    # x = x.repeat().batch(lotes)

    historial = model.fit(x,y, epochs=20, steps_per_epoch= math.ceil(len(x)/lotes))

    return  historial.history['loss']



def modelo_denso_2():
    print('Ejecutando modelo Denso 2')
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape= (30,30,3)),
        tf.keras.layers.Dense(200,activation = tf.nn.relu),
        tf.keras.layers.Dense(50,activation = tf.nn.relu),
        tf.keras.layers.Dense(50,activation = tf.nn.relu),
        tf.keras.layers.Dense(200,activation = tf.nn.relu),
        tf.keras.layers.Dense(116,activation = tf.nn.softmax),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    lotes = 32
    # x = x.repeat().batch(lotes)

    historial = model.fit(x,y, epochs=20, steps_per_epoch= math.ceil(len(x)/lotes))
    return  historial.history['loss']

resul_convolucional = modelo_convolucional()
resul_denso_1 = modelo_denso_1()
resul_denso_2 = modelo_denso_2()


plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(resul_convolucional,label = 'convolucional')
plt.plot(resul_denso_1,label = 'Denso 1')
plt.plot(resul_denso_2,label = 'Denso 2')
plt.legend()


plt.show()

# model.save('./modelo/modelo.h5')
# model.save_weights('./modelo/pesos.h5')
