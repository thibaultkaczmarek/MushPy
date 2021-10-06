import pandas as pd
import time
start=time.time()
import numpy as np
from tqdm import tqdm
import itertools # Pour créer des iterateurs
from PIL import Image
import os

import matplotlib.pyplot as plt  # Pour l'affichage d'images
from matplotlib import cm # Pour importer de nouvelles cartes de couleur

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential, load_model

#################################################################################################################
# Epoch 10/10
# 137/137 [==============================] - 399s 3s/step - loss: 1.6010 - accuracy: 0.2377 - val_loss: 1.6022 - val_accuracy: 0.2365
#################################################################################################################

csv_5fam = "reduced_dataset_5_families.csv"
df = pd.read_csv(csv_5fam)

folderpath = "C:/Users/Sevil/Desktop/datascientest/projet/local/all_data_set/"

df['filepath'] = [folderpath + str(df['image_id'][x]) + ".jpg" for x in range(len(df))]

data_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
print(data_train.head())

train_data_generator = ImageDataGenerator(rotation_range=5,
                                          width_shift_range = 0.1,
                                          height_shift_range = 0.05,
                                          zoom_range = 1.1)
test_data_generator = ImageDataGenerator()

batch_size = 100
data_train["label"] = data_train["label"].astype(str)
data_test["label"] = data_test["label"].astype(str)

train_generator = train_data_generator.flow_from_dataframe(dataframe = data_train, directory="", x_col = "filepath", y_col="label", target_size=(256,256), batch_size=batch_size, class_mode='sparse')
test_generator = test_data_generator.flow_from_dataframe(dataframe = data_test, directory="", x_col = "filepath", y_col="label", target_size=(256,256), batch_size=batch_size, class_mode='sparse')


lenet=Sequential()
lenet.add(Conv2D(filters=30,input_shape=(256,256,3),kernel_size=(5,5),activation="relu"))
lenet.add(MaxPooling2D(pool_size=(2,2)))
lenet.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu"))
lenet.add(MaxPooling2D(pool_size=(2,2)))
lenet.add(Dropout(0.2))
lenet.add(Flatten())
lenet.add(Dense(128,activation="relu"))
lenet.add(Dense(5,activation="softmax"))


lenet.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                         patience = 3,
                                         factor = 0.5,
                                         verbose = 2,
                                         mode = 'min')
early_stopping = callbacks.EarlyStopping(monitor = "val_loss",
                                         patience = 3,
                                         mode = 'min',
                                         verbose = 2,
                                         restore_best_weights= True)
checkpoint = callbacks.ModelCheckpoint(filepath="C:/Users/Sevil/Desktop/datascientest/projet/local/model_lenet",
                                      monitor = 'val_loss',
                                      save_best_only = True,
                                      save_weights_only = False,
                                      mode = 'min',
                                      save_freq = 'epoch')

history = lenet.fit(train_generator, 
                              epochs = 10,
                              steps_per_epoch = len(data_train)//batch_size,
                              validation_data = test_generator,
                              validation_steps = len(data_test)//batch_size,
                              callbacks=[lr_plateau, early_stopping, checkpoint])

save_name = "C:/Users/Sevil/Desktop/datascientest/projet/local/model_lenet"
lenet.save(save_name)



plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.title('Model loss by epoch')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
plt.show()


