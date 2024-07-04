import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import random
import cv2

test_dir = './Data/ASL_dataset3/Test_Alphabet'
model_dir = 'Data/models/PROJECT104.keras'

classi = [folder[len(test_dir) + 1:] for folder in glob.glob(test_dir + '/*')]
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

grafico_classi(test_dir)

# caricamento del modello da testare
model = tf.keras.models.load_model(model_dir)
print(model.summary())

# normalizzazione dei valori
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

accuracy = model.evaluate(test_generator)
print(f"Accuracy: {accuracy[1] * 100:.2f}%")

y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

#creazione della Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predette')
plt.ylabel('Effettive')
plt.title('Confusion Matrix')
plt.show()

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot di precision, recall e F1 score
metrics = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
plt.xlabel('Metriche')
plt.ylabel('Valori')
plt.title('Precision, Recall, F1 Score')
plt.show()