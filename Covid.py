'''
detect Covid with radiology of chest using VGG16
Data downloaded from https://www.kaggle.com/datasets/alifrahman/covid19-chest-xray-image-dataset

alifathi8008@gmail.com

4/14/2022
'''

import tensorflow as tf
import numpy as np
import glob 
from tensorflow.keras import layers as ksl
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2 as cv
# Load the data
def LoadData(addr):
    imgs = []
    for img in glob.glob(addr + "/*"):
        img = cv.imread(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (224, 224))
        img = img / 255.0
        imgs.append(img)
    return np.array(imgs)
Covid = LoadData(r"CovidDataSet/dataset/covid")
Normal = LoadData(r"CovidDataSet/dataset/normal")
Data = np.concatenate((Covid, Normal))
Label = np.array([1] * len(Covid) + [0] * len(Normal))
Label = tf.keras.utils.to_categorical(Label, num_classes=2)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.2, random_state=42)
# Data Augmentation
Gen = ImageDataGenerator(rotation_range=15,
                         width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         shear_range=0.1, 
                         zoom_range=0.1, 
                         horizontal_flip=True, 
                         fill_mode="nearest")
# Create the model
BaseModel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential()
model.add(BaseModel)
model.add(ksl.Flatten())
model.add(ksl.Dense(512, activation='relu'))
model.add(ksl.Dropout(0.5))
model.add(ksl.Dense(2, activation='softmax'))

for i in BaseModel.layers:
    i.trainable = False

model.summary()
# Compile the model
model.compile(optimizer=Adam(lr=0.0001,decay=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
Hist = model.fit(Gen.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train)//32, validation_data=[X_test,y_test], epochs=7)
# save model
model.save(r"Model/Covid.h5")

# plot History
plt.plot(Hist.history['accuracy'])
plt.plot(Hist.history['val_accuracy'])
plt.title('model accuracy')
plt.savefig(r'Plots/accuracy.png')
plt.show()
plt.plot(Hist.history['loss'])
plt.plot(Hist.history['val_loss'])
plt.title('model loss')
plt.savefig(r'Plots/loss.png')
plt.show()