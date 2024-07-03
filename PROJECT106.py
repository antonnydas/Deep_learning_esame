import matplotlib.pyplot as plt
import numpy as np
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Conv2D, MaxPooling2D
from keras.src.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import random
import cv2
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras import models
import tensorflow as tf

tf.random.set_seed(0)

train_dir = './Data/ASL_dataset3/Train_Alphabet'

# definizione dei parametri in ingresso
img_size = 512
num_classi = 27
target_size = (64, 64)
target_dims = (64, 64, 3)
batch_size = 32
validation_split = 0.2

# Creazione dei generatori
# preprocessing dei dati e normalizzazione, data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,

                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=40,
                                   validation_split=validation_split
                                   )

val_datagen = ImageDataGenerator(rescale=1. / 255,
                                 validation_split=validation_split)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    seed=0,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    seed=0,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

# visualizzaizone di un sample per classe
classi = [folder[len(train_dir) + 1:] for folder in glob.glob(train_dir + '/*')]
classi.sort()


def grafico_classi(base_path):
    colonne = 5
    righe = int(np.ceil(len(classi) / colonne))
    fig = plt.figure(figsize=(16, 20))

    for i in range(len(classi)):
        cls = classi[i]
        img_path = base_path + '/' + cls + '/**'
        path_contents = glob.glob(img_path)

        immagini = random.sample(path_contents, 1)

        sp = plt.subplot(righe, colonne, i + 1)
        plt.imshow(cv2.imread(immagini[0])[:, :, ::-1])  # Convert BGR to RGB
        plt.title(cls)
        sp.axis('off')

    plt.show()


grafico_classi(train_dir)

model = models.Sequential([

    Conv2D(128, (3, 3), activation='relu', input_shape=target_dims),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(512, (3, 3), activation='relu'),
    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(num_classi, activation='softmax')

])

print(model.summary())

model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# lr scheduler per adattare il learning rate in base alle epoche
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# early stopping
model_es = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True)

# training del modello
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[lr_scheduler, model_es]
)

# Plot con i risultati del training
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy del modello')
plt.ylabel('Accuracy')
plt.xlabel('Epoche')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

model.save('PROJECT106.keras')
