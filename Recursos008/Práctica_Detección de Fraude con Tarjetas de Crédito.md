# Práctica

### Detección de Fraude con Tarjetas de Crédito.

### 1. **Preparación del Entorno y Carga del Dataset**

Comienza con la instalación de las librerías necesarias y carga del dataset desde Kaggle.

```python
# Instalación de librerías en Google Colab (si no están instaladas)
!pip install -q pandas scikit-learn matplotlib seaborn

# Importación de librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. **Limpieza y Organización de los Datos**

Este paso es crucial para identificar y manejar valores nulos, escalar las características y balancear las clases.

```python
# Carga de datos
data = pd.read_csv('/path/to/creditcard.csv')  # Reemplaza con la ruta del archivo
print("Datos cargados con éxito")
data.info()

# Revisión de valores nulos
data.isnull().sum()

# Escalado de datos
from sklearn.preprocessing import StandardScaler
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# División entre características y etiquetas
X = data.drop('Class', axis=1)
y = data['Class']
```

### 3. **División del Dataset y Entrenamiento**

Dividimos los datos en conjuntos de entrenamiento y prueba, luego entrenamos el modelo.

```python
# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Modelo entrenado con éxito")
```

### 4. **Evaluación del Modelo con Métricas y Gráficos**

Cada métrica se explicará y se visualizará para comprender mejor el rendimiento del modelo.

### A. **Matriz de Confusión**

La matriz de confusión es una herramienta visual que muestra los verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos de las predicciones del modelo. Ayuda a comprender cuántas transacciones fraudulentas y no fraudulentas fueron clasificadas correctamente o incorrectamente.

- **Verdaderos Positivos (TP)**: Transacciones que eran fraudulentas y el modelo predijo como fraudulentas.
- **Falsos Positivos (FP)**: Transacciones que no eran fraudulentas, pero el modelo las predijo como fraudulentas (falsos positivos).
- **Verdaderos Negativos (TN)**: Transacciones que no eran fraudulentas y el modelo predijo como no fraudulentas.
- **Falsos Negativos (FN)**: Transacciones que eran fraudulentas, pero el modelo no las detectó (falsos negativos).

Visualizar la matriz ayuda a ver los errores del modelo y cómo podría ajustarse para reducirlos.

```python
# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraude", "Fraude"], yticklabels=["No Fraude", "Fraude"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
```



### B. **Exactitud (Accuracy)**

La exactitud mide el porcentaje de predicciones correctas (tanto de fraude como de no fraude). Sin embargo, en un problema de clases desbalanceadas, como el fraude, la exactitud puede ser engañosa: un modelo podría clasificar casi todas las transacciones como “no fraude” y obtener una exactitud alta, pero no detectar fraudes reales.

**Fórmula**:
$$
Exactitud = ( TP + TN )  /  ( TP + TN + FP + FN )
$$

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud: {accuracy}")
```



### C. **Precisión y Exhaustividad (Recall)**

Estas métricas evalúan el desempeño del modelo específicamente en la clase positiva (fraude).

- **Precisión (Precision)**: Mide la proporción de transacciones que el modelo predijo como fraudes y que realmente lo eran. Es clave para evitar falsos positivos y se usa mucho en problemas donde el coste de clasificar erróneamente algo como positivo es alto.

  **Fórmula**:
  $$
  Precision = ( TP ) / ( TP + FP )
  $$
  

  **Exhaustividad (Recall)**: Mide la capacidad del modelo para encontrar todos los fraudes reales, sin importar si predijo algunos falsos positivos. Es importante cuando queremos reducir el riesgo de perder fraudes, incluso si eso significa detectar algunos falsos positivos.

  **Fórmula**:

  
  $$
  Recall = ( TP ) / ( TP + FN )
  $$
  

```python
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precisión: {precision}")
print(f"Exhaustividad: {recall}")
```



### D. **F1-Score**

El F1-Score combina precisión y recall en una sola métrica mediante la media armónica. Es particularmente útil en problemas de clases desbalanceadas como el fraude, ya que penaliza fuertemente los modelos que no tienen precisión y recall altos a la vez.

**Fórmula**:


$$
F1 - Score = 2 * ((Precision * Recall) / (Precision * Recall))
$$


```python
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1}")
```



### E. **Curva ROC y AUC (Área Bajo la Curva)**

La **Curva ROC (Receiver Operating Characteristic)** muestra la tasa de verdaderos positivos frente a la tasa de falsos positivos en diferentes umbrales de decisión, permitiéndonos ver cómo el modelo discrimina entre clases. El **AUC (Área Bajo la Curva ROC)** mide el desempeño global del modelo en la discriminación entre las clases, siendo 1 un modelo perfecto y 0.5 uno que predice al azar.

- **Tasa de Verdaderos Positivos (True Positive Rate, TPR)**: Es el recall.
- **Tasa de Falsos Positivos (False Positive Rate, FPR)**: Mide el porcentaje de falsos positivos sobre todas las instancias negativas.

**Curva ROC**: Una curva ROC que se aproxima a la esquina superior izquierda indica un buen desempeño.

```python
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



### F. **Especificidad (Specificity)**

La especificidad mide cuán bien el modelo identifica correctamente las transacciones no fraudulentas, es decir, la capacidad de evitar falsos positivos.

**Fórmula**:
$$
Especificidad = ( TN ) / ( TN + FP )
$$


Una alta especificidad es deseable en sistemas donde es importante no clasificar erróneamente transacciones normales como fraude, ya que puede ser molesto para los clientes.

```python
specificity = tn / (tn + fp)
print(f"Especificidad: {specificity}")
```



## **5. Aplicación: Validación de Fraude**

Simula una interfaz en línea de comandos, en la que puedes ingresar manualmente los valores de una transacción para que el modelo determine si es sospechosa o no.

#### -  **Importación de Librerías y Preparación de Datos**

Primero, asegúrate de que el modelo esté entrenado siguiendo los pasos anteriores. Aquí asumimos que ya tienes el modelo entrenado y los datos preparados. Además, si no has hecho antes la división en características y etiquetas (`X_train`, `X_test`, `y_train`, `y_test`), hazlo ahora.

#### **- Función de Validación de Fraude**

La función `verificar_fraude` tomará una lista de valores de transacción como entrada, la transformará para que sea compatible con el modelo y luego usará el modelo entrenado para hacer una predicción.

```python
def verificar_fraude(transaccion):
    """
    Función para verificar si una transacción es fraudulenta o no.
    
    Parámetros:
    - transaccion: Una lista con los valores de una transacción específica en el mismo orden que las columnas de X_test.
    
    Retorna:
    - Un mensaje indicando si se detecta fraude o no.
    """
    # Verificar que la entrada tiene el tamaño correcto
    if len(transaccion) != X_test.shape[1]:
        return "Error: La transacción debe tener exactamente {} valores.".format(X_test.shape[1])
    
    # Convertir la entrada en un arreglo de numpy con las dimensiones adecuadas
    transaccion = np.array(transaccion).reshape(1, -1)
    
    # Realizar predicción con el modelo
    prediccion = model.predict(transaccion)
    
    # Interpretar el resultado
    if prediccion[0] == 1:
        return "⚠️ Fraude Detectado: Esta transacción parece sospechosa."
    else:
        return "✅ No hay fraude detectado: Esta transacción parece segura."
```

#### **- Simulación de Entrada de Datos en Google Colab**

Para probar esta función, podemos simular una transacción ingresando manualmente los valores. Aquí tienes un ejemplo de cómo pedir entrada para los valores.

```python
# Ejemplo de uso de la función verificar_fraude
# Los valores ingresados deben estar en el mismo orden que las columnas de X_test

# Simulación de una transacción
print("Ingrese los valores de la transacción en el mismo orden de las columnas de X_test")

# Por ejemplo, el primer valor podría ser el tiempo (segundos) desde la primera transacción registrada
valores = [
    float(input("Tiempo: ")),
    float(input("V1: ")),
    float(input("V2: ")),
    # Continúa con el resto de los valores...
    float(input("Amount (monto de la transacción): "))
]

# Validación de fraude
resultado = verificar_fraude(valores)
print(resultado)
```

#### **- Interfaz Simple para Pruebas en Google Colab**

Para una experiencia más sencilla, podrías ofrecer una transacción de ejemplo con valores predeterminados para que la aplicación sea fácil de probar sin tener que ingresar datos cada vez.

```python
# Valores de ejemplo para probar
transaccion_ejemplo = X_test.iloc[0].tolist()  # Usa una transacción real del conjunto de prueba

# Ejecuta la función de verificación de fraude en esta transacción de ejemplo
print("Transacción de ejemplo:")
print(verificar_fraude(transaccion_ejemplo))
```

### 5. **Explicación del Código**

- **Función `verificar_fraude`**: Esta función verifica el tamaño de los datos ingresados, convierte la entrada en un arreglo de numpy y utiliza el modelo para predecir. Devuelve un mensaje adecuado según el resultado de la predicción.
- **Entrada de datos en Colab**: Dado que Colab no ofrece una interfaz gráfica avanzada, utilizamos `input()` para recibir los valores de cada campo. Este enfoque es funcional para simulaciones y permite validar rápidamente.
- **Prueba rápida**: Puedes usar el método `iloc` de pandas para tomar una transacción directamente del conjunto de prueba y pasarla a la función `verificar_fraude`.

















