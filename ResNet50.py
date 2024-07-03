import matplotlib.pyplot as plt
import numpy as np
from keras.src.applications.resnet import ResNet50
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import random
import cv2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras import models
import tensorflow as tf

tf.random.set_seed(0)

train_dir = './Data/ASL_dataset3/Train_Alphabet'

# definizione dei parametri in ingresso
img_size = 512
num_classi = 27
target_size = (224, 224)
target_dims = (224, 224, 3)
batch_size = 32
validation_split = 0.2

# Creazione dei generatori
# preprocessing dei dati e normalizzazione, data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   rotation_range=40,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.2,
                                   zoom_range=0.4,
                                   brightness_range=(0.5, 1.5),
                                   fill_mode='nearest',
                                   horizontal_flip=True,
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

# importazione della Resnet
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# fine tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(128, activation='relu'),
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

model.save('PROJECT104.keras')
