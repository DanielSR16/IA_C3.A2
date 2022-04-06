from tensorflow import keras
import cv2 
import tensorflow as tf
import numpy as np
new_model = keras.models.load_model("modelo/modelo.h5")

image = cv2.resize(cv2.imread(f'elote.jpg'),(30,30))
# print(image )
probar_img = tf.cast(image,tf.float32)
probar_img /= 255
array_img = np.asarray([probar_img])

predic = new_model.history["loss"]
print(predic)

# print(predic)
# res = name_clas[np.argmax(predic[0])]
# print(np.argmax(predic[0]))
# print(res)
