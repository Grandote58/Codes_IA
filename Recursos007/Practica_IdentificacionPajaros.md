Clasificación de aves usando Python



#### **Objetivo:**

Entrenar un modelo de clasificación para identificar si un ave pertenece a una de las siguientes especies:

- **Jilguero americano**
- **Lechuza común**
- **Abejaruco carmín**
- **Pájaro carpintero velloso**
- **Pingüino Emperador**
- **Flamenco**

El dataset utilizado proviene de [Kaggle: Bird Species Dataset](https://www.kaggle.com/datasets/rahmasleam/bird-speciees-dataset), y entrenaremos un modelo utilizando **Regresión Logística**. Evaluaremos el modelo con las métricas: **Matriz de Confusión, Exactitud, Precisión, Recall, F1-Score, y la Curva ROC**. Además, implementaremos una funcionalidad para cargar imágenes y realizar predicciones con el modelo.

### **Pasos a seguir:**

------

### Paso 1: **Configurar Google Colab y Cargar el Dataset**

1. **Descarga y carga del conjunto de datos** desde Kaggle:

   - En tu entorno de Google Colab, primero instala Kaggle para poder descargar el dataset.

   ```bash
   !pip install kaggle
   ```

2. **Sube el archivo kaggle.json** (que contiene las credenciales de tu cuenta de Kaggle) a tu entorno de Colab para acceder al dataset. Luego, descarga el dataset desde Kaggle:

   ```bash
   # Crear una carpeta .kaggle y mover el archivo kaggle.json
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   
   # Descargar el dataset desde Kaggle
   !kaggle datasets download -d rahmasleam/bird-speciees-dataset
   
   # Descomprimir el dataset
   !unzip bird-speciees-dataset.zip
   ```

------

### Paso 2: **Carga y Exploración de Datos**

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos (imágenes) y las etiquetas
# En este ejemplo, asumimos que las imágenes están organizadas en carpetas según la especie
#/content/Bird Speciees Dataset
data_dir = "/content/Bird Speciees Dataset"

# Crear un DataFrame para las rutas de las imágenes y las etiquetas
image_paths = []
labels = []

# Recorrer las carpetas del dataset y agregar las rutas y etiquetas
for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, image_name))
            labels.append(label)

# Crear el DataFrame con las rutas y las etiquetas
df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.head())

# Ver la distribución de clases
sns.countplot(data=df, x="label")
plt.xticks(rotation=45)
plt.show()
```

------

### Paso 3: **Preprocesamiento de Imágenes y Etiquetas**

Vamos a utilizar `tensorflow` para cargar y preprocesar las imágenes.

```python
import tensorflow as tf

# Definir tamaño de imagen y batch size
image_size = (150, 150)
batch_size = 32

# Codificar las etiquetas
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Cargar y preprocesar imágenes
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalización
    return image

# Crear un dataset de TensorFlow
def load_image_and_label(row):
    image = preprocess_image(row['image_path'])
    label = row['label_encoded']
    return image, label

# Convertir DataFrame a dataset
image_dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], df['label_encoded']))
image_dataset = image_dataset.map(lambda x, y: (preprocess_image(x), y))

# Dividir en conjunto de entrenamiento y prueba
train_size = int(0.8 * len(df))
train_dataset = image_dataset.take(train_size).batch(batch_size)
test_dataset = image_dataset.skip(train_size).batch(batch_size)
```

------

### Paso 4: **Entrenamiento del Modelo de Clasificación (Regresión Logística)**

Aunque TensorFlow es generalmente usado para redes neuronales, aquí entrenaremos un modelo de **Regresión Logística** usando una capa densa en la salida.

```python
from tensorflow.keras import layers, models

# Crear un modelo simple con capas de convolución y una capa densa para la clasificación
model = models.Sequential([
    layers.InputLayer(input_shape=(150, 150, 3)),  # Tamaño de imagen (150x150 con 3 canales RGB)
    layers.Conv2D(32, (3, 3), activation='relu'),  # Primera capa de convolución
    layers.MaxPooling2D((2, 2)),                   # Pooling para reducir tamaño
    layers.Conv2D(64, (3, 3), activation='relu'),  # Segunda capa de convolución
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),                              # Aplanar la salida para conectarla a la capa densa
    layers.Dense(64, activation='relu'),           # Capa densa intermedia
    layers.Dense(6, activation='softmax')          # Capa de salida con 6 clases (una para cada especie)
])

# Compilar el modelo (definir el optimizador, la función de pérdida y las métricas)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)
```

------

### Paso 5: **Evaluación del Modelo con Métricas**

Evaluamos el modelo utilizando las métricas clave:

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import numpy as np

# Predecir en el conjunto de prueba
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# Matriz de Confusión
matriz_confusion = confusion_matrix(y_true, y_pred)
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.ylabel("Etiquetas Reales")
plt.xlabel("Predicciones")
plt.show()

# Reporte de Clasificación (Precisión, Recall, F1)
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1], pos_label=1)
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.show()

```

------

### Paso 6: **Subir una Imagen y Realizar Predicción**

Finalmente, implementamos una funcionalidad donde puedes subir una imagen y hacer una predicción con el modelo.

```python
from google.colab import files
from PIL import Image

# Subir imagen
uploaded = files.upload()

# Preprocesar la imagen subida
for filename in uploaded.keys():
    img = Image.open(filename)
    img = img.resize(image_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Añadir dimensión batch

    # Realizar la predicción
    pred = model.predict(img)
    pred_class = label_encoder.inverse_transform([np.argmax(pred)])
    print(f"La imagen subida probablemente pertenece a: {pred_class[0]}")
```

------

### Conclusión:

En esta práctica hemos entrenado y evaluado un modelo de clasificación para identificar diferentes especies de aves, utilizando imágenes de un dataset y aplicando métricas clave como la matriz de confusión, precisión, recall, F1-Score y la curva ROC. También implementamos la funcionalidad para subir una imagen y obtener una predicción del modelo entrenado.

Este flujo de trabajo puede adaptarse a otros problemas de clasificación utilizando diferentes datasets y modelos. ¡Sigue experimentando con los datos y ajusta el modelo según tus necesidades!