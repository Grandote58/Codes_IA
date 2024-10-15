# Aprendizaje Supervisado con TensorFlow y Breast Cancer Dataset

------

##### **1. Recolección y Carga de Datos**

- El primer paso en el aprendizaje supervisado es **cargar los datos**. Utilizaremos el **Breast Cancer Dataset** disponible en Kaggle, que contiene características relacionadas con tumores que pueden ser benignos o malignos.
- **Datos**: Cada fila del dataset es una observación (muestra de células) con atributos numéricos como el grosor de los grumos, tamaño celular, cromatina, etc., y una columna de "etiqueta" que indica si el tumor es benigno o maligno.

##### **2. Preprocesamiento de los Datos**

- Antes de entrenar el modelo, los datos deben ser limpiados y normalizados para garantizar que el algoritmo funcione correctamente. El preprocesamiento incluye:
  - Remover datos nulos o incompletos.
  - Dividir los datos en características (X) y etiquetas (y).
  - Estandarizar los valores para que todos los atributos tengan la misma escala.

##### **3. Definir la Arquitectura del Modelo**

- En este paso, **se define la estructura del modelo** de aprendizaje supervisado. Utilizaremos un **Perceptrón Multicapa (MLP)**, una red neuronal simple que es ideal para problemas de clasificación.
- La arquitectura incluye capas de entrada, capas ocultas (con funciones de activación) y una capa de salida.

##### **4. Entrenamiento del Modelo**

- El modelo se entrena utilizando los datos etiquetados (entrenamiento supervisado). Aquí ajustamos los parámetros del modelo para minimizar el error y mejorar la precisión en la predicción.

##### **5. Evaluación del Modelo**

- Después de entrenar, evaluamos el rendimiento del modelo utilizando un **conjunto de prueba** separado para medir su precisión. Se utilizarán métricas como la **exactitud (accuracy)**.

  

------

### **Práctica Guiada: Flujo de Trabajo en el Aprendizaje Supervisado**

#### **1. Instalación y Configuración del Entorno**

Para esta práctica, utilizaremos **Visual Studio Code** y **Python**. Asegúrate de tener instaladas las siguientes librerías:

##### **a. Crear un entorno virtual e instalar librerías**

1. **Crear entorno virtual** (recomendado para gestionar dependencias):

   ```bash
   python -m venv venv
   ```

2. **Activar el entorno virtual**:

   - En Windows: `venv\Scripts\activate`
   - En Mac/Linux: `source venv/bin/activate`

3. **Instalar las librerías necesarias**:

   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib
   ```

------

#### **2. Estructura del Proyecto**

Organiza los archivos de tu proyecto de la siguiente manera:

```bash
breast_cancer_project/
│
├── venv/  # Entorno virtual
│
├── data/  # Carpeta de datos
│   └── breast-cancer.csv  # Dataset descargado de Kaggle
│
├── models/  # Carpeta para almacenar el modelo entrenado
│   └── mlp_model.h5
│
├── train_model.py  # Código de entrenamiento
├── README.md  # Descripción del proyecto
└── requirements.txt  # Dependencias del proyecto
```



#### **Código para Cargar y Preprocesar los Datos**

El siguiente código carga y preprocesa el dataset para preparar los datos para el modelo:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os

# Deshabilitar las optimizaciones de oneDNN para evitar advertencias de rendimiento
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar el dataset
data = pd.read_csv('data/breast-cancer.csv')

# Mostrar las primeras filas para verificar los datos
print(data.head())

# Dividir el dataset en características (X) y etiquetas (y)
X = data.drop(columns=['id', 'diagnosis'])  # Excluir las columnas 'id' y 'diagnosis'
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convertir 'M' en 1 y 'B' en 0

# Dividir los datos en conjuntos de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos para que todas las características estén en la misma escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Explicación del código:**

- El dataset se carga desde un archivo CSV.
- Se eliminan las columnas no necesarias y se convierten las etiquetas de texto ('M' para maligno y 'B' para benigno) en valores binarios.
- Se dividen los datos en conjuntos de entrenamiento (70%) y prueba (30%).
- Se normalizan los datos utilizando `StandardScaler` para que los valores estén en la misma escala.



#### **Definir la Arquitectura del Modelo**

En este paso, definimos la arquitectura del modelo utilizando **TensorFlow** y la API de **Keras**. El modelo será un Perceptrón Multicapa (MLP):

```python
# Definir el modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Capa de entrada
    Dense(32, activation='relu'),  # Capa oculta
    Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo (20 épocas)
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test))

# Guardar el modelo entrenado
model.save('models/mlp_model.h5')
```

**Explicación del código:**

- Se crea un modelo secuencial con capas densamente conectadas.
- La capa de entrada tiene 64 neuronas, la oculta tiene 32, y la de salida usa una activación `sigmoid` para predicciones binarias (0 o 1).
- Se compila el modelo usando el optimizador `adam` y la función de pérdida `binary_crossentropy`.
- El modelo se entrena por 20 épocas con un tamaño de lote de 32.



#### **5. Evaluar la Precisión del Modelo**

Ahora evaluamos la precisión del modelo en el conjunto de prueba:

```python
# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_scaled, y_test)

print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")

# Graficar la precisión durante el entrenamiento
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión del modelo durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

```

**Explicación del código:**

- Se evalúa el modelo en los datos de prueba para obtener la precisión.
- El resultado muestra el porcentaje de predicciones correctas en el conjunto de prueba.

------

#### **6. Iniciar el Proyecto en Visual Studio Code**

1. Asegúrate de estar en el entorno virtual (actívalo si es necesario).

2. Corre el archivo de entrenamiento:

   ```python
   python train_model.py
   ```

El modelo se entrenará y se guardará en la carpeta `models/` con el nombre `mlp_model.h5`.



### **Probar el Modelo con Nuevas Muestras**

A continuación, tienes un código detallado que sigue estos pasos:

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Deshabilitar las optimizaciones de oneDNN para evitar advertencias de rendimiento
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Cargar el modelo entrenado
model = tf.keras.models.load_model('models/mlp_model.h5')

# Cargar el dataset original para obtener el escalador
data = pd.read_csv('data/breast-cancer.csv')

# Preprocesamiento: Asegurarnos de que solo usemos 30 características
# 'diagnosis' es la etiqueta que queremos predecir, y 'id' no es relevante
X = data.drop(columns=['id', 'diagnosis'])  # Eliminamos 'id' y 'diagnosis'
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convertimos 'M' en 1 y 'B' en 0

# Verificar que las características de entrada sean 30
print(f"Número de características: {X.shape[1]}")  # Debe imprimir 30

# Normalización de las características de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Ajustar el escalador con los datos originales

# Nuevas muestras con exactamente 30 características (valores ficticios)
nuevas_muestras = np.array([[10, 5, 10, 6, 8, 4, 7, 3, 9, 1, 10, 5, 10, 6, 8, 4, 7, 3, 9, 1,
                             10, 5, 10, 6, 8, 4, 7, 3, 9, 1]])  # Ejemplo con 30 características

# Asegurarse de que las nuevas muestras tengan exactamente 30 características
if nuevas_muestras.shape[1] != 30:
    raise ValueError("Las nuevas muestras deben tener exactamente 30 características.")

# Escalar las nuevas muestras con el mismo escalador usado en el entrenamiento
nuevas_muestras_scaled = scaler.transform(nuevas_muestras)

# Realizar la predicción con el modelo
prediccion = model.predict(nuevas_muestras_scaled)

# Convertir la predicción en un valor legible (benigno o maligno)
resultado = 'Maligno' if prediccion[0] > 0.5 else 'Benigno'
print(f"El tumor es: {resultado}")

```

### **Explicación del Código:**

1. **Carga del Modelo**: Utilizamos `tf.keras.models.load_model()` para cargar el modelo previamente guardado.
2. **Preprocesamiento de Nuevas Muestras**:
   - Creamos una nueva muestra de datos con características como grosor del bulto, uniformidad de tamaño celular, etc.
   - Esta muestra es escalada usando el mismo `StandardScaler` que se aplicó a los datos de entrenamiento.
3. **Predicción**: El modelo predice si la nueva muestra representa un tumor benigno o maligno. La predicción es un número entre 0 y 1. Si el número es mayor a 0.5, el tumor es clasificado como maligno, de lo contrario, es benigno.
4. **Resultado**: La predicción se convierte en una etiqueta legible ('Maligno' o 'Benigno') y se imprime el resultado.











