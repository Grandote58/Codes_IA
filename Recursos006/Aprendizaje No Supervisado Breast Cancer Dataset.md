# **Flujo de Trabajo en el Aprendizaje No Supervisado**

Este flujo de trabajo se enfoca en aplicar técnicas de aprendizaje no supervisado para identificar patrones ocultos en los datos sin utilizar etiquetas. Utilizaremos el algoritmo **K-Means** para realizar **clustering** en el *Breast Cancer Dataset* de Kaggle.

------

### **1. Instalación de Librerías**

Primero, asegúrate de que tienes las bibliotecas necesarias instaladas para trabajar en un entorno de Python. Si no las tienes, puedes instalar las dependencias usando los siguientes comandos:

#### Crear entorno virtual y activarlo:

```bash
python -m venv venv
```

En Windows:

```bash
venv\Scripts\activate
```

En Linux/Mac:

```bash
source venv/bin/activate
```

#### Instalar las bibliotecas necesarias:

```bash
pip install pandas scikit-learn matplotlib
```

------

### **2. Estructura de Archivos**

Organiza tu proyecto de la siguiente manera:

```css
BreastCancer_Clustering/
│
├── data/  # Carpeta donde almacenaremos el dataset
│   └── breast-cancer.csv
│
├── models/  # Carpeta para almacenar los modelos entrenados
│
├── clustering_model.py  # Script principal para entrenar el modelo
├── test_clustering.py  # Script para probar el modelo y visualizar los resultados
├── requirements.txt  # Dependencias del proyecto
└── README.md  # Documentación del proyecto
```

------

### **3. Práctica: Entrenamiento del Modelo con K-Means**

En esta práctica, utilizaremos el algoritmo K-Means para agrupar los datos de pacientes sin utilizar etiquetas, buscando patrones en los datos.

#### **3.1. Preprocesamiento de Datos**

Crea el archivo `clustering_model.py` para entrenar el modelo K-Means:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Cargar el dataset
data = pd.read_csv('data/breast-cancer.csv')

# Preprocesamiento: eliminar las columnas 'id' y 'diagnosis'
X = data.drop(columns=['id', 'diagnosis'])

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Guardar el modelo entrenado
if not os.path.exists('models'):
    os.makedirs('models')

# Guardar los centros del clustering
centers = kmeans.cluster_centers_
pd.DataFrame(centers).to_csv('models/kmeans_centers.csv', index=False)

print("Entrenamiento completado. El modelo K-Means ha sido entrenado y los centros han sido guardados.")
```

#### **3.2. Documentación del Código**

- **Cargar y Preprocesar los Datos**: Se elimina la columna 'id' (que no aporta valor en el clustering) y 'diagnosis' (que es la etiqueta en un problema supervisado). Las características se normalizan para que todas tengan la misma escala.
- **K-Means**: Se define el algoritmo K-Means para agrupar los datos en 2 clusters (benignos y malignos, aunque no utilizamos etiquetas aquí).
- **Guardar el Modelo**: Se guarda la información de los centros de los clusters en un archivo CSV para poder reutilizar el modelo más adelante.

#### **3.3. Evaluar los Resultados Visualmente**

Crea un archivo `test_clustering.py` para visualizar los resultados del modelo:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el dataset y los modelos
data = pd.read_csv('data/breast-cancer.csv')

# Preprocesamiento: eliminar las columnas 'id' y 'diagnosis'
X = data.drop(columns=['id', 'diagnosis'])

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cargar el modelo K-Means previamente entrenado
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Asignar cada muestra a un cluster
data['cluster'] = kmeans.labels_

# Visualización de los clusters (usamos las dos primeras características para graficar)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['cluster'], cmap='viridis')
plt.title('Clustering con K-Means')
plt.xlabel('Característica 1 (normalizada)')
plt.ylabel('Característica 2 (normalizada)')
plt.show()
```

#### **Explicación del Código**:

1. **Cargar el Dataset**: Se carga el dataset de cáncer de mama y se preprocesan los datos eliminando las columnas 'id' y 'diagnosis'.
2. **Normalización de Características**: Se normalizan los datos para asegurar que todas las características estén en la misma escala.
3. **Asignar Clusters**: Cada muestra se agrupa en uno de los dos clusters.
4. **Visualización**: Se usa un gráfico de dispersión para visualizar cómo se han agrupado las muestras en base a las dos primeras características.

------

### **4. Script para Probar el Modelo**

Este script permite ver cómo el modelo K-Means agrupa los datos sin tener etiquetas. Puedes visualizar los clusters y analizar los patrones descubiertos.

#### **4.1. Ejecutar el Modelo**

Para entrenar y probar el modelo, sigue estos pasos en la terminal:

```python
# Entrenar el modelo K-Means
python clustering_model.py

# Probar el modelo y visualizar los clusters
python test_clustering.py
```

------

### **5. Explicación del Aprendizaje No Supervisado con K-Means**

**K-Means** es una técnica de clustering que agrupa datos sin utilizar etiquetas, buscando patrones ocultos. En este caso, agrupa las muestras de pacientes en dos clusters, representando potencialmente grupos de células benignas y malignas, sin usar la etiqueta `diagnosis`. El algoritmo ajusta las posiciones de los centroides iterativamente hasta minimizar la distancia de los puntos de datos a sus centroides más cercanos.



### **Script Completo: Simulación y Prueba del Modelo K-Means**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Simular nuevas características de un paciente basado en las mismas características del dataset original
# Simulamos 30 características que representen valores típicos de un paciente
nuevos_pacientes = np.array([
    [10, 5, 10, 6, 8, 4, 7, 3, 9, 1, 10, 5, 10, 6, 8, 4, 7, 3, 9, 1,
     10, 5, 10, 6, 8, 4, 7, 3, 9, 1],  # Paciente 1
    [7, 3, 5, 4, 6, 2, 4, 2, 5, 2, 7, 3, 5, 4, 6, 2, 4, 2, 5, 2,
     7, 3, 5, 4, 6, 2, 4, 2, 5, 2]    # Paciente 2
])

# Cargar el dataset original para obtener el escalador y los parámetros de clustering
data = pd.read_csv('data/breast-cancer.csv')

# Preprocesar los datos originales
X = data.drop(columns=['id', 'diagnosis'])  # Eliminar 'id' y 'diagnosis'
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalizar las características

# Definir y entrenar el modelo K-Means con los datos originales
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Normalizar las características de los nuevos pacientes con el mismo escalador
nuevos_pacientes_scaled = scaler.transform(nuevos_pacientes)

# Usar el modelo K-Means para predecir a qué cluster pertenecen los nuevos pacientes
predicciones = kmeans.predict(nuevos_pacientes_scaled)

# Mostrar las predicciones
for i, prediccion in enumerate(predicciones):
    resultado = 'Posible cáncer' if prediccion == 1 else 'Probablemente no cáncer'
    print(f"Paciente {i+1}: {resultado}")

# Visualización de los nuevos pacientes en los clusters existentes
# Graficamos las dos primeras características de los pacientes simulados y los datos originales
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', label='Datos originales')
plt.scatter(nuevos_pacientes_scaled[:, 0], nuevos_pacientes_scaled[:, 1], color='red', marker='X', label='Nuevos pacientes')
plt.title('Clustering de Nuevos Pacientes con K-Means')
plt.xlabel('Característica 1 (normalizada)')
plt.ylabel('Característica 2 (normalizada)')
plt.legend()
plt.show()
```

### **Explicación del Código:**

1. **Simulación de Nuevos Pacientes**:
   - Hemos simulado dos nuevos pacientes, cada uno con 30 características, que representan las mismas propiedades que el dataset original (por ejemplo, grosor del bulto, uniformidad de tamaño celular, etc.).
2. **Preprocesamiento**:
   - Al igual que en el modelo entrenado, los datos de los nuevos pacientes son **normalizados** usando el mismo `StandardScaler` que se ajustó con el conjunto de datos original.
3. **K-Means Clustering**:
   - Se utiliza el modelo K-Means previamente entrenado para predecir a qué cluster pertenecen los nuevos pacientes. El modelo agrupa los datos en dos clusters, que podrían representar pacientes con cáncer y pacientes sin cáncer, aunque esta clasificación es realizada sin utilizar etiquetas.
4. **Predicciones**:
   - Basado en el cluster asignado, el script imprime si el paciente pertenece al cluster que podría ser "Posible cáncer" (cluster 1) o "Probablemente no cáncer" (cluster 0).
5. **Visualización**:
   - Se genera una gráfica de dispersión donde los nuevos pacientes se marcan en rojo con un marcador "X", sobre los datos originales agrupados en los clusters. Esto permite ver visualmente dónde se ubican los nuevos pacientes en relación a los clusters existentes.

------

### **Cómo Ejecutar el Script**

1. Guarda el script en un archivo llamado `simulate_kmeans.py` dentro de tu proyecto.
2. Asegúrate de tener el dataset en la carpeta `data/` como se menciona en la estructura de archivos.
3. Ejecuta el script desde la terminal o desde tu entorno de desarrollo:

```python
python simulacion.py
```