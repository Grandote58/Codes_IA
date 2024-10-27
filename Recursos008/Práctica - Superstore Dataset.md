# Práctica - Superstore Dataset

Esta practica te ayudará al supermercado a analizar qué productos, regiones, categorías y segmentos de clientes tienen el mejor rendimiento para optimizar su estrategia de ventas.

https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

### Objetivo de la Práctica

El objetivo es ayudar a un supermercado a entender mejor su data para decidir en qué productos, regiones, categorías y segmentos de clientes enfocarse o evitar. También se incluye un modelo de predicción que evalúa el rendimiento de diferentes características en la detección de patrones en los datos de ventas.

### Pasos a seguir:

### 1. **Preparación del Entorno y Carga del Dataset**

Primero, instalamos las librerías necesarias y cargamos el dataset desde Kaggle.

```python
# Instalación de librerías necesarias en Google Colab
!pip install -q pandas scikit-learn matplotlib seaborn

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
```

### 2. **Limpieza y Organización de los Datos**

En esta etapa, identificamos y eliminamos datos irrelevantes y realizamos el procesamiento de variables categóricas. También verificamos los valores nulos.

```python
# Carga de datos
data = pd.read_csv('/path/to/Superstore.csv')  # Cambia la ruta con tu archivo en Google Colab

# Información inicial sobre los datos
data.info()
data.describe()

# Limpieza de datos
# Eliminación de columnas irrelevantes
data.drop(columns=['Order ID', 'Customer ID', 'Product ID'], inplace=True)

# Revisión de valores nulos
data.isnull().sum()

# Transformación de variables categóricas a numéricas (ejemplo usando get_dummies)
data = pd.get_dummies(data, drop_first=True)
```

### 3. **Análisis Exploratorio de Datos (EDA)**

Este paso nos ayuda a entender las ventas en cada categoría, región y segmento, proporcionando una mejor perspectiva para determinar las áreas de enfoque.

```python
# Análisis de ventas por categoría
plt.figure(figsize=(10,5))
sns.barplot(x='Category', y='Sales', data=data, estimator=sum)
plt.title("Ventas por Categoría")
plt.show()

# Análisis de ventas por región
plt.figure(figsize=(10,5))
sns.barplot(x='Region', y='Sales', data=data, estimator=sum)
plt.title("Ventas por Región")
plt.show()

# Análisis de ventas por segmento de cliente
plt.figure(figsize=(10,5))
sns.barplot(x='Segment', y='Sales', data=data, estimator=sum)
plt.title("Ventas por Segmento de Cliente")
plt.show()
```

### 4. **Preparación para el Modelo Predictivo**

Creamos una variable objetivo para clasificar si una transacción es rentable o no, basándonos en los beneficios.

```python
# Creación de la variable objetivo
data['Rentable'] = np.where(data['Profit'] > 0, 1, 0)

# Separación de características (X) y variable objetivo (y)
X = data.drop(columns=['Rentable', 'Sales', 'Profit'])
y = data['Rentable']

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. **Entrenamiento del Modelo**

Entrenamos un modelo de clasificación para predecir la rentabilidad de una transacción.

```python
# Entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Modelo entrenado con éxito")
```

### 6. **Evaluación del Modelo con Métricas y Gráficos**

#### 6.1. **Matriz de Confusión**

```python
# Predicciones del modelo
y_pred = model.predict(X_test)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Rentable", "Rentable"], yticklabels=["No Rentable", "Rentable"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
```

#### 6.2. **Exactitud, Precisión, Recall, F1-Score y Curva ROC**

Calculamos las métricas para evaluar el modelo:

```python
# Exactitud
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud: {accuracy}")

# Precisión y Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precisión: {precision}")
print(f"Exhaustividad: {recall}")

# F1-Score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1}")

# Curva ROC y AUC
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.show()
```

### 7. **Aplicación para Validar Rentabilidad**

Creamos una aplicación para verificar si una transacción es rentable o no:

```python
def verificar_rentabilidad(transaccion):
    """
    Función para verificar si una transacción es rentable.
    
    Parámetros:
    - transaccion: Lista con valores en el mismo orden que las columnas de X_test.
    
    Retorna:
    - Un mensaje indicando si es rentable o no.
    """
    # Verificación de entrada
    if len(transaccion) != X_test.shape[1]:
        return "Error: La transacción debe tener {} valores.".format(X_test.shape[1])
    
    # Transformar entrada y predecir
    transaccion = np.array(transaccion).reshape(1, -1)
    prediccion = model.predict(transaccion)
    
    # Resultado
    return "✅ Rentable" if prediccion[0] == 1 else "❌ No Rentable"
```

#### Ejemplo de Uso

Prueba la función con una transacción de ejemplo:

```python
# Ejemplo de transacción
transaccion_ejemplo = X_test.iloc[0].tolist()

# Verificar rentabilidad
print(verificar_rentabilidad(transaccion_ejemplo))
```