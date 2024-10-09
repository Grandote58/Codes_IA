# **Proyecto Completo: Aplicación de Laberinto con Agente y Paredes Aleatorias Usando Q-Learning**

En este proyecto, vamos a crear una aplicación de laberinto donde un agente navega por un entorno con paredes generadas aleatoriamente. La interfaz permitirá seleccionar el tamaño del laberinto a través de una caja de texto desplegable (combo box), que generará un laberinto cuadrado con paredes distribuidas aleatoriamente. El agente se moverá a través del laberinto, y la ruta será visualizada utilizando **matplotlib**.

### **Estructura del Proyecto**

```
bashCopiar código/laberinto_qlearning_project
│
├── main.py              # Archivo principal que ejecuta la aplicación
├── laberinto.py         # Función para generar el laberinto con paredes aleatorias
├── qlearning.py         # Simulación del agente navegando el laberinto
├── interfaz.py          # Interfaz gráfica con CustomTkinter
├── rutas.agente.txt     # Alamcenara la informacion de la rutas
└── README.md            # Documentación del proyecto
```

------

### **Paso 1: Configuración del entorno de trabajo**

1. Asegúrate de tener Python instalado y **Visual Studio Code** configurado.

2. Crea un entorno virtual:

   ```bash
   python -m venv env
   source env/bin/activate   # Para Linux/Mac
   .\env\Scripts\activate    # Para Windows
   ```

3. Instala las librerías necesarias:

   ```bash
   pip install numpy matplotlib customtkinter
   ```



------

### **Paso 2: Implementar las Funciones**

#### 3.1. Archivo: `laberinto.py`

Este archivo contiene la función para generar el laberinto con paredes aleatorias.

```python
# laberinto.py

import numpy as np
import random

def generar_laberinto(tamano):
    """
    Genera un laberinto cuadrado con paredes aleatorias.
    
    Parámetros:
        tamano (int): El tamaño del laberinto (filas y columnas).
    
    Retorna:
        numpy.ndarray: El laberinto generado con caminos (0) y paredes (1).
    """
    laberinto = np.zeros((tamano, tamano), dtype=int)

    # Establecer la posición inicial y la salida
    laberinto[0, 0] = 0  # Inicio
    laberinto[tamano-1, tamano-1] = 0  # Meta

    # Rellenar con paredes aleatorias
    for i in range(tamano):
        for j in range(tamano):
            if (i, j) != (0, 0) and (i, j) != (tamano-1, tamano-1):
                # Probabilidad de 30% de que sea una pared
                laberinto[i, j] = 1 if random.random() < 0.3 else 0

    return laberinto

```

------

#### 3.2. Archivo: `qlearning.py`

Este archivo implementa la simulación del agente que navega por el laberinto generado.

```python
import numpy as np

'''
def simular_agente(laberinto, max_rutas=5):
    """
    Simula hasta 'max_rutas' posibles rutas óptimas en el laberinto, usando un enfoque más eficiente.
    
    Parámetros:
        laberinto (numpy.ndarray): El laberinto generado.
        max_rutas (int): Número máximo de rutas a generar (por defecto, 5).
    
    Retorna:
        list: Lista de rutas generadas.
    """
    rutas = []  # Lista para almacenar las rutas generadas
    
    # Dimensiones del laberinto
    max_filas, max_columnas = laberinto.shape
    acciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Derecha, Abajo, Izquierda, Arriba (más eficientes)

    # Generar 'max_rutas' diferentes rutas
    for _ in range(max_rutas):
        estado = [0, 0]  # Posición inicial del agente
        ruta = [estado.copy()]  # Inicializar la ruta con la posición inicial
        visitados = set()  # Mantener un registro de las posiciones visitadas
        visitados.add(tuple(estado))

        for _ in range(100):  # Limitar a 100 pasos por ruta
            posibles_movimientos = []
            
            # Evaluar las posibles acciones para moverse hacia la meta
            for accion in acciones:
                siguiente_estado = [estado[0] + accion[0], estado[1] + accion[1]]

                # Verificar si el siguiente estado está dentro de los límites y no es una pared
                if (0 <= siguiente_estado[0] < max_filas and 
                    0 <= siguiente_estado[1] < max_columnas and
                    laberinto[siguiente_estado[0], siguiente_estado[1]] == 0 and
                    tuple(siguiente_estado) not in visitados):
                    posibles_movimientos.append(siguiente_estado)
            
            # Si hay movimientos posibles, elegir el más prometedor (hacia la meta)
            if posibles_movimientos:
                # Ordenar los movimientos posibles por proximidad a la meta (meta es [max_filas-1, max_columnas-1])
                posibles_movimientos.sort(key=lambda pos: abs(max_filas - 1 - pos[0]) + abs(max_columnas - 1 - pos[1]))
                estado = posibles_movimientos[0]  # Elegir el movimiento que más acerca a la meta
                visitados.add(tuple(estado))  # Marcar como visitado
            else:
                break  # Si no hay movimientos posibles, detener la ruta

            ruta.append(estado.copy())  # Añadir la nueva posición a la ruta

            # Si el agente llega a la meta, terminamos esta ruta
            if estado == [max_filas - 1, max_columnas - 1]:
                break
        
        rutas.append(ruta)  # Almacenar la ruta generada

    return rutas

 '''


def es_valido(laberinto, pos, visitados):
    """
    Verifica si una posición es válida dentro del laberinto (no es una pared, no está fuera de límites y no ha sido visitada).
    
    Parámetros:
        laberinto (numpy.ndarray): El laberinto.
        pos (tuple): La posición actual del agente (fila, columna).
        visitados (set): Conjunto de posiciones ya visitadas.
    
    Retorna:
        bool: True si la posición es válida, False en caso contrario.
    """
    filas, columnas = laberinto.shape
    x, y = pos
    return 0 <= x < filas and 0 <= y < columnas and laberinto[x, y] == 0 and pos not in visitados

def buscar_rutas(laberinto, pos, meta, ruta_actual, rutas, visitados):
    """
    Realiza un backtracking para encontrar todas las rutas posibles del agente hasta la meta.
    
    Parámetros:
        laberinto (numpy.ndarray): El laberinto.
        pos (tuple): La posición actual del agente (fila, columna).
        meta (tuple): La posición de la meta (fila, columna).
        ruta_actual (list): La ruta actual que el agente está siguiendo.
        rutas (list): Lista para almacenar todas las rutas posibles.
        visitados (set): Conjunto de posiciones ya visitadas.
    """
    # Si el agente ha alcanzado la meta, añadir la ruta a la lista de rutas posibles
    if pos == meta:
        rutas.append(ruta_actual.copy())
        return

    visitados.add(pos)  # Marcar la posición actual como visitada

    # Movimientos posibles: Derecha, Abajo, Izquierda, Arriba
    movimientos = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for dx, dy in movimientos:
        nueva_pos = (pos[0] + dx, pos[1] + dy)
        if es_valido(laberinto, nueva_pos, visitados):
            ruta_actual.append(nueva_pos)
            buscar_rutas(laberinto, nueva_pos, meta, ruta_actual, rutas, visitados)
            ruta_actual.pop()  # Deshacer el movimiento (backtracking)

    visitados.remove(pos)  # Desmarcar la posición actual para otras posibles rutas

def simular_agente(laberinto):
    """
    Genera todas las posibles rutas del agente desde el inicio hasta la meta, y devuelve la ruta más corta.
    
    Parámetros:
        laberinto (numpy.ndarray): El laberinto generado.
    
    Retorna:
        list: La ruta óptima (más corta) encontrada, o una lista vacía si no hay ruta.
    """
    rutas = []  # Almacenará todas las rutas posibles
    inicio = (0, 0)  # Posición inicial
    meta = (laberinto.shape[0] - 1, laberinto.shape[1] - 1)  # Meta (última celda)
    
    # Realizar backtracking para encontrar todas las rutas posibles
    buscar_rutas(laberinto, inicio, meta, [inicio], rutas, set())
    
    if rutas:
        # Encontrar la ruta más corta
        ruta_optima = min(rutas, key=len)
        return ruta_optima
    else:
        print("No se encontró ninguna ruta válida.")
        return []  # Si no se encontró ninguna ruta

```

------

#### 3.3. Archivo: `interfaz.py`

Este archivo contiene la interfaz gráfica donde el usuario puede seleccionar el tamaño del laberinto.

```python
import customtkinter as ctk
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from laberinto import generar_laberinto
from qlearning import simular_agente

# Variables globales para gestionar rutas y tiempo
rutas = []
tiempo_restante = 60
detenido_por_tiempo = False
temporizador_activo = False  # Nueva variable para controlar si el temporizador está activo

def visualizar_ruta_optima(laberinto, ruta_optima):
    """
    Visualiza el laberinto con la ruta óptima pintada en naranja.
    
    Parámetros:
        laberinto (numpy.ndarray): El laberinto generado.
        ruta_optima (list): La ruta óptima generada por el agente.
    """
    if not ruta_optima:
        print("No se puede visualizar la ruta porque no se encontró ninguna ruta óptima.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    # Definir el mapa de colores para el laberinto (caminos: blanco, paredes: negro)
    cmap = colors.ListedColormap(['white', 'black'])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Dibujar el laberinto
    ax.imshow(laberinto, cmap=cmap, norm=norm)

    # Dibujar los pasos de la ruta óptima en naranja
    for idx, (x, y) in enumerate(ruta_optima):
        ax.plot(y, x, 'o-', color='orange', markersize=8 - idx * 0.05)

    # Marcar el inicio y la meta
    ax.plot(0, 0, 'bo', markersize=10, label='Inicio')
    ax.plot(laberinto.shape[0] - 1, laberinto.shape[1] - 1, 'go', markersize=10, label='Meta')

    # Añadir título y mostrar
    ax.set_title(f"Ruta Óptima (Pasos: {len(ruta_optima)})", fontsize=14)
    plt.show()

def guardar_rutas(rutas):
    """
    Guarda las rutas del agente en un archivo de texto.
    
    Parámetros:
        rutas (list): Lista de rutas generadas por el agente.
    """
    with open("rutas_agente.txt", "w") as archivo:
        for ruta in rutas:
            for paso in ruta:
                archivo.write(f"{paso} ")
            if tuple(ruta[-1]) == (len(ruta) - 1, len(ruta) - 1):
                archivo.write("ruta aprobada")
            archivo.write("\n")

def mostrar_ultimos_graficos():
    """Muestra los últimos tres gráficos si el tiempo se agota."""
    if len(rutas) > 0:
        ultimas_rutas = rutas[-3:]  # Obtener las últimas tres rutas generadas
        for idx, ruta in enumerate(ultimas_rutas):
            plt.figure(figsize=(5, 5))
            laberinto = generar_laberinto(len(ruta))  # Generar laberinto con el tamaño adecuado
            visualizar_ruta_optima(laberinto, ruta)
            plt.title(f"Último gráfico {idx+1}")
            plt.show()
    else:
        print("No hay rutas para mostrar.")

def actualizar_contador(label_contador, ventana):
    """
    Actualiza el contador de tiempo cada segundo.
    """
    global tiempo_restante, detenido_por_tiempo, temporizador_activo
    if tiempo_restante > 0 and temporizador_activo:  # Solo si el temporizador está activo
        tiempo_restante -= 1
        label_contador.configure(text=f"Tiempo restante: {tiempo_restante} segundos")
        ventana.after(1000, lambda: actualizar_contador(label_contador, ventana))
    elif tiempo_restante == 0:
        detenido_por_tiempo = True
        temporizador_activo = False
        print("Tiempo agotado. Mostrando últimos gráficos.")
        mostrar_ultimos_graficos()

def iniciar_simulacion(label_contador, ventana):
    """
    Inicia la simulación del agente dentro del laberinto.
    """
    global tiempo_restante, detenido_por_tiempo, temporizador_activo, rutas
    tiempo_restante = 60  # Reiniciar el tiempo cada vez que se inicia la simulación
    detenido_por_tiempo = False  # Reiniciar el estado del tiempo
    temporizador_activo = True  # Activar el temporizador
    rutas = []  # Reiniciar las rutas

    # Obtener el tamaño del laberinto seleccionado por el usuario
    tamano_seleccionado = int(combobox_tamano.get())
    laberinto = generar_laberinto(tamano_seleccionado)
    rutas_generadas = simular_agente(laberinto)

    # Si no se ha detenido por tiempo, agregamos la ruta generada
    if not detenido_por_tiempo:
        rutas.append(rutas_generadas)
        if rutas_generadas:  # Si se encontró una ruta válida
            visualizar_ruta_optima(laberinto, rutas_generadas)
        else:
            print("No se encontró ninguna ruta válida.")
    else:
        print("Simulación detenida debido al límite de tiempo.")

def cerrar_aplicacion(ventana):
    """
    Cierra la aplicación, guarda las rutas y detiene el temporizador.
    """
    global temporizador_activo
    guardar_rutas(rutas)
    temporizador_activo = False  # Detener el temporizador
    ventana.quit()

def crear_interfaz():
    """
    Crea la interfaz gráfica donde el usuario puede seleccionar el tamaño del laberinto.
    """
    global combobox_tamano

    ventana = ctk.CTk()
    ventana.title("Generador de Laberinto")
    ventana.geometry("600x300")

    label_tamano = ctk.CTkLabel(ventana, text="Selecciona el tamaño del laberinto (5 a 15):")
    label_tamano.pack(pady=10)

    combobox_tamano = ctk.CTkComboBox(ventana, values=[str(i) for i in range(5, 16)])
    combobox_tamano.pack(pady=10)

    # Etiqueta para mostrar el tiempo restante
    label_contador = ctk.CTkLabel(ventana, text=f"Tiempo restante: {tiempo_restante} segundos")
    label_contador.pack(pady=10)

    boton_iniciar = ctk.CTkButton(ventana, text="Generar Laberinto", command=lambda: iniciar_simulacion(label_contador, ventana))
    boton_iniciar.pack(pady=20)

    # Botón para cerrar la aplicación
    boton_cerrar = ctk.CTkButton(ventana, text="Cerrar Aplicación", command=lambda: cerrar_aplicacion(ventana))
    boton_cerrar.pack(pady=20)

    # Iniciar el contador
    actualizar_contador(label_contador, ventana)

    ventana.mainloop()

```

### **Explicaciones Clave:**

1. **Control del Temporizador (`temporizador_activo`)**:
   - Se añadió la variable **`temporizador_activo`** para controlar si el temporizador está en funcionamiento.
   - Al hacer clic en **Cerrar Aplicación**, el temporizador se detiene, evitando que continúe restando tiempo innecesariamente.
2. **Reinicio de la Simulación**:
   - Al hacer clic en **Iniciar Simulación**, se reinician las rutas y el temporizador, garantizando que se pueda iniciar una nueva simulación desde cero.
   - Si el usuario hace clic en **Iniciar Simulación** varias veces, cada simulación empezará desde el principio.
3. **Cierre de la Aplicación**:
   - **Guardar Rutas**: Las rutas se guardan en el archivo `rutas_agente.txt` antes de cerrar la aplicación.
   - **Detener el Temporizador**: El temporizador se desactiva al cerrar la aplicación para que no siga corriendo en segundo plano.

### **Uso Completo del Código**:

1. **Iniciar Simulación**: Cuando el usuario hace clic en **Iniciar Simulación**, el temporizador se activa, se generan las rutas, y si se vuelve a hacer clic, todo se reinicia.
2. **Cerrar Aplicación**: Al hacer clic en **Cerrar Aplicación**, las rutas se guardan y el temporizador se detiene.
3. **Temporizador**: Si el tiempo se agota, el temporizador se detiene y se muestran los últimos tres gráficos generados.



------

#### 3.4. Archivo: `main.py`

Este es el archivo principal que inicia la aplicación.

```python
# main.py

from interfaz import crear_interfaz

if __name__ == "__main__":
    crear_interfaz()
```