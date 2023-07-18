import pathlib, os, random, mplcyberpunk
import numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from glob import glob
from skimage.io import imread
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import load_img, img_to_array
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, MaxPool2D, Conv2D, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential, Model
from keras import optimizers
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from mlxtend.plotting import plot_confusion_matrix
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint

wandb.init (project = 'k-ium23', entity = 'binhyc11')
model_callback = ModelCheckpoint (filepath='D:/models/I_A.h5', save_weights_only=False,
                                    monitor='val_loss', mode='min', save_best_only=True)

base_path = 'D:/k-ium/traindataset_3/I/A/'

batch_size = 32
img_height, img_width = 100, 100
lr_rate = 1e-3
epoch = 1000
input_shape = (img_height, img_width, 3)

datagen= ImageDataGenerator(rescale=1/255, validation_split=0.3)

train_data = datagen.flow_from_directory(base_path, target_size=(img_height, img_width),
                                      batch_size=batch_size, class_mode='categorical', shuffle=False, subset='training')

val_data = datagen.flow_from_directory(base_path, target_size=(img_height, img_width),
                                      batch_size=batch_size, class_mode='categorical', shuffle=False, subset='validation')

class_name=train_data.class_indices

class_names=list(class_name.keys())

num_classes=len(class_name.keys())

model = Sequential()
model.add (Conv2D(256, (9, 9), activation='relu',
                  kernel_initializer= 'he_uniform',
                  padding = 'same',
                  input_shape = input_shape
                  ))
model.add (MaxPooling2D(3, 3))

# model.add (Conv2D(128, (7, 7), activation='relu'))
# model.add (MaxPooling2D(2, 2))

model.add (Conv2D(64, (5, 5), activation='relu'))
model.add (MaxPooling2D(3, 3))

# model.add (Conv2D(128, (3, 3), activation='relu'))
# model.add (MaxPooling2D(2, 2))

model.add (Flatten())
model.add (Dense(128, activation= 'relu', kernel_initializer= 'he_uniform'))

model.add(Dropout(0.5))

model.add (Dense(num_classes, activation= 'softmax'))

opt = keras.optimizers.Adam (learning_rate = lr_rate)
model.compile(optimizer=opt, loss = 'categorical_crossentropy', 
              metrics=['accuracy'])

# callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
history= model.fit(train_data,
                    steps_per_epoch= train_data.samples//train_data.batch_size,
                    validation_data= val_data,
                    validation_steps= val_data.samples//val_data.batch_size,
                    epochs= epoch,
                    verbose=1,
                    callbacks = [model_callback ,WandbCallback()]
                  )

prob_predict= model.predict(val_data, steps=np.ceil(val_data.samples/val_data.batch_size), verbose=1)
class_predict = [np.argmax(t) for t in prob_predict]


