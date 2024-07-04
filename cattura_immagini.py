import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from rembg import remove
from PIL import Image


# directories dei modelli
model_dir = './Data/models/ResNet50.keras'
model = tf.keras.models.load_model(model_dir)

# parametri in ingresso
img_height, img_width = 224,224
num_classi = 27

class_labels = [
    'A', 'B','vuoto', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z'
]


# Preprocessing delle immagini e rimozione dello sfondo
def preprocess_image(image):

    image = Image.fromarray(image)
    image = remove(image)
    image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (img_height, img_width))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    preprocessed_frame = preprocess_image(frame)

    prediction = model.predict(preprocessed_frame)
    predicted_class = class_labels[np.argmax(prediction)]


    cv2.putText(frame, f'Lettera: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255,255, 255), 1, cv2.LINE_AA)
    cv2.imshow('RICONOSCIMENTO DELL''ASL', frame)

    # Fine del ciclo nel momento in cui viene premuto q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
