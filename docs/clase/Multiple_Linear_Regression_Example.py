#-------------------------------------------------------------------------------
# Name:        Multiple_Linear_Regression_Example
# Purpose:      El objetiva es predecir el crecimiento total en mm. de
#               una planta de albahaca a partir de determinadas variables
#               de ambiente y contextuales. Se simula un Dataset de 100
#               muestras aleatorias, pero respetando las condiciones reales
#               del ambiente y del tipo de planta. Para disminuir la tasa
#               de error, puede probarse setear más muestras.
#
# Author:      Simón Polizzi
#
# Created:     03/10/2024
# Copyright:   (c) Simón Polizzi 2024
# Licence:     Simón Polizzi
#-------------------------------------------------------------------------------

import pandas as pd # se usa para manipulación de datos (DataFrames, series).
import numpy as np # para operaciones numéricas, generación de datos aleatorios y arrays.
import matplotlib.pyplot as plt # para graficar datos en 2D
from mpl_toolkits.mplot3d import Axes3D # habilita la proyección 3D dentro de matplotlib.
from sklearn.linear_model import LinearRegression # para crear y entrenar un modelo de regresión lineal múltiple.

# --- --- --- MÉTRICAS
from sklearn.model_selection import train_test_split # splitear los features (etiquetas de dato de entrenamiento)
                                                    # y dividir datos de prueba "test" y datos de entrenamiento "train"
from sklearn.metrics import r2_score # calcula el R2 a partir del entrenamiento resultante anteriormente

def main(): # programa base/principal
    # Generar datos aleatorios que tengan lógica dentro del contexto del problema
    # ----------------------------
    np.random.seed(42) # fija la semilla aleatoria para reproducibilidad. indica el patrón
                        # en el que se van generando los valores. puede seleccionar
                        # más valores que otros dependiendo del 'seed'.
    n = 100 # número de muestras/datos a generar.
    data = pd.DataFrame({ # crea un DataFrame, tabla de datos con columnas: Sol, Agua, Estacion, Compost.
        "Sol": np.random.uniform(4, 12, n),           # hs. sol/día --> genera n valores aleatorios uniformes entre a y b.
        "Agua": np.random.uniform(200, 1500, n),      # ml. de agua/semanal
        "Estacion": np.random.choice(["Primavera","Verano","Otoño","Invierno"], n), # genera n valores aleatorios seleccionados de una lista.
        "Compost": np.random.choice(["Sí","No"], n)
    })

    # Función de crecimiento (heurística)
    # ----------------------------
    def crecimiento(row): # función que calcula el crecimiento de cada planta según variables.
                            # 'row' es un objeto 'Series' de 'Pandas' que contiene los valores de una fila.
        base = 50 # Se asigna directamente un valor constante de 50.
                  # Esto representa un crecimiento base inicial, independientemente de sol, agua o compost.
        sol = 60 * row["Sol"] # row["Sol"]/row["Agua"] --> valores de la fila correspondiente a las columnas llamadas 'Sol' y 'Agua'
                            # del DataFrame.
                            # row es una fila del DataFrame que se pasa a la función crecimiento.
                            # row["Sol"] devuelve el valor de la columna "Sol" de esa fila.
        agua = (6 * row["Agua"]) / 10 # 'sol' y 'agua' --> contribución del sol y agua al crecimiento.

        # Ajuste por estación
        # Ajusta el crecimiento según la estación y condiciones del sol, agua y compost.
        if (row["Estacion"] == "Primavera"):
            est = 50
        elif (row["Estacion"] == "Verano"):
            est = 50 if row["Sol"] <= 7 else -30
            est = est + (-20) if (row["Agua"] <= 800 and row["Sol"] > 7) else 30
        elif (row["Estacion"] == "Otoño"):
            est = -70
            if (row["Compost"] == "Sí"):
                est += 20
            if (row["Sol"] >= 7):
                est += 30
            if (row["Agua"] >= 1000):
                est += 30
            if (row["Compost"]=="No" or row["Sol"]<=6 or row["Agua"]<=500):
                est -= 35
        else:  # Invierno
            est = -95

        # bonificación extra por usar compost.
        comp = 25 if (row["Compost"]=="Sí") else 0

        # Ruido aleatorio
        ruido = np.random.normal(0, 5) # genera un ruido aleatorio con distribución normal (gaussiana).
                                        # Ruido en este contexto significa una pequeña variación impredecible
                                        # que agregamos a los cálculos.
                                        # Sirve para que los datos no sean perfectamente predecibles.
                                        # En la vida real, no todas las plantas crecen exactamente igual aunque
                                        # tengan las mismas condiciones: eso lo simulamos con ruido.
                                        # La mayoría de los valores generados están cerca de la media.
                                        # Hay menos valores muy altos o muy bajos.
                                        # Forma una campana cuando dibujas un histograma (de ahí
                                        # el nombre “curva de campana” o "bell curve").
                                        # 0: la media, porque no queremos que el ruido aumente o disminuya sistemáticamente el crecimiento.
                                        # 5: desviación estándar, controla cuánto puede variar el crecimiento por azar.
        return (base + sol + agua + est + comp + ruido) # Suma todas las contribuciones para devolver el crecimiento final estimado.

    # data.apply(funcion, axis=1) --> aplica la función crecimiento fila por fila del DataFrame.
    # axis=1 --> indica que se aplica a lo largo de las filas (recorrido horizontal).
    # Se crea la columna "Crecimiento" en el DataFrame.
    # En pandas, el eje 0 o 1 se refiere a cómo se recorre la tabla.
    # .apply() toma cada fila completa y pasa como 'Series' a la función.
    data["Crecimiento"] = data.apply(crecimiento, axis=1)


    # Codificar variables categóricas
    # ----------------------------
    # pd.get_dummies() convierte categorías en columnas binarias (0/1).
    # Cada fila tendrá 1 si pertenece a esa categoría, 0 si no.
    # convierte texto en columnas binarias para poder usarlo en la regresión.
    # drop_first=True evita que haya columnas redundantes
    # (por ejemplo, si todas las demás son 0, sabemos que es la categoría faltante).
    #  columns=["Estacion","Compost"] --> solo se codifican estas columnas.
    data_encoded = pd.get_dummies(data, columns=["Estacion","Compost"], drop_first=True)

    # Separar X e y
    # Cada columna es un array de números (float o int).
    # devuelve un nuevo DataFrame sin la columna "Crecimiento".
    # Esto es porque X representa las variables independientes que el modelo usará para aprender.
    X = data_encoded.drop(columns=["Crecimiento"])
    # selecciona solo la columna "Crecimiento".
    # Extrae la columna "Crecimiento" como un 'Pandas Series' (array unidimensional).
    y = data_encoded["Crecimiento"]
    # tenemos como resultado lo siguiente:
    # X: matriz 2D (n_filas × n_features) [mXn] → para el modelo.
    # y: vector 1D (n_filas) [1Xm] → lo que el modelo intenta predecir.
    # X: entradas con todas las variables explicativas.
    # y: salida que queremos estimar.
    # Si no separamos, el modelo no sabría cuál columna predecir y cuál usar como predictor.

    # Entrenar modelo de regresión lineal múltiple
    # ----------------------------
    model = LinearRegression() # crea un objeto de regresión lineal múltiple de 'sklearn'.
                                # Python prepara estructuras internas para almacenar
                                # coeficientes (coef_) e intercepto (intercept_).
    model.fit(X, y) # 'model' es un objeto que contiene métodos como fit(), predict() y atributos como coef_.
                    # método que entrena el modelo.
                    # Qué hace internamente: Toma X (matriz de features, dimensiones n_filas × n_variables) y Y como (vector objetivo).
                    # Busca los coeficientes βi (pendientes) y el intercepto β₀ que minimicen el error cuadrático.
                    # Calcula los valores óptimos usando álgebra lineal.
                    # Guarda los resultados en: model.coef_ --> array con coeficientes de cada variable (β₁, β₂, ...).
                                                # model.intercept_ --> valor del intercepto (β₀).

    print("Modelo entrenado. Coeficientes:")
    # X.columns --> lista de nombres de las variables (por ejemplo: ["Sol", "Agua", "Estacion_Otoño", ...]).
    # model.coef_ --> coeficientes que indican cuánto influye cada variable en el crecimiento
    # pd.DataFrame({...}) --> crea un DataFrame para mostrarlo de manera legible.
    # El coeficiente con mayor valor absoluto tiene mayor influencia directa sobre la variable de salida.
    # Cada hora extra de sol aumenta el crecimiento en ~59 mm
    # En primavera, el crecimiento promedio aumenta 120 mm respecto a invierno
    print(pd.DataFrame({"Variable": X.columns, "Coeficiente": model.coef_}))

    # Predecir un nuevo valor
    # ----------------------------
    # len(X.columns): es el número de columnas/features que usó el modelo al entrenar.
    # np.zeros((1, len(...))): crea una matriz 2D con 1 fila y tantas columnas como features,
    # rellenada con ceros.
    # pd.DataFrame(..., columns=X.columns) transforma esa matriz en un DataFrame
    # con exactamente las mismas columnas y el mismo orden que X.
    # al comenzar con ceros aseguras de que todas las columnas que el modelo
    # espera existan y estén inicializadas. Luego modificas solo las de interés.
    nuevo_valor = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    nuevo_valor["Sol"] = 11 # asignación de valores/características
    nuevo_valor["Agua"] = 1200
    nuevo_valor["Estacion_Otoño"] = 1
    nuevo_valor["Compost_Sí"] = 1

    prediccion = model.predict(nuevo_valor)[0] # retorna un número (float) que es la predicción del crecimiento para esos valores
                                                # pasados como Diccionario. aunque solo haya una predicción,
                                                # el resultado sigue siendo un array con un elemento. con [0] solo tomamos el 1ro.
    print(f"Predicción de crecimiento para el nuevo valor: {prediccion:.2f} mm")

    # dividir en entrenamiento y prueba --> 30% de datos para test y el otro 70% para train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    #   predecir sobre el conjunto de prueba --> predecir si el modelo se ajusta lo suficientemente bien
    # con datos que nunca vio ni utilizó para el proceso de train del algoritmo. debe devolver una Y'..
    y_pred = model.predict(X_test)
    # calcular coeficiente --> R2 de la regresión.. a partir de la comparación de las predicciones realizadas con
    # datos que jamas analizo.. compara y_test que es el dataset original de X_test.. con la nueva y_pred.. es decir..
    # los resultados estimados con datos que no uso para su entrenamiento
    r2 = r2_score(y_test, y_pred)
    print(f"Coeficiente de determinación (R²): {r2:.2f}")
    print(f"Porcentaje de certeza del modelo: {r2 * 100:.2f}%")

    # comparar predicción con el promedio general
    promedio_crecimiento = data["Crecimiento"].mean()  # calcula el promedio de la columna Crecimiento
    print(f"Promedio general de crecimiento: {promedio_crecimiento:.2f} mm")
    # Comparar y mostrar resultado
    if (prediccion > promedio_crecimiento):
        print("El crecimiento predicho es MAYOR que el promedio general.")
    elif (prediccion < promedio_crecimiento):
        print("El crecimiento predicho es MENOR que el promedio general.")
    else:
        print("El crecimiento predicho es IGUAL al promedio general.")


    # gráfico 3D con matplotlib incluyendo nuevo valor
    # ----------------------------
    fig = plt.figure(figsize=(10,8)) # plt.figure: crea una nueva figura (como una hoja en blanco
                                    # figsize: define el tamaño de la figura en pulgadas (10 de ancho, 8 de alto).
    ax = fig.add_subplot(111, projection='3d') # agregando un eje tridimensional (es decir,
                                            # el sistema de coordenadas X, Y, Z) dentro de esa figura.
                                            # '111': “código de posición” que indica cómo organizar los subgráficos dentro de la figura.
                                            # projection='3d': Por defecto, matplotlib crea gráficos 2D.
                                            # Con este argumento, le decís que el eje será tridimensional. activa el modo 3D.

    # Colores como diccionario para cada estación del año
    colores = {"Primavera":"green", "Verano":"orange", "Otoño":"red", "Invierno":"blue"}

    # Símbolos como diccionario para el estado del compostaje
    simbolos = {"Sí":"o", "No":"^"}

    # Graficar datos originales
    for estacion in data["Estacion"].unique(): # Selecciona la columna "Estacion" del DataFrame.
                                                # Devuelve una lista con los valores únicos que aparecen en esa columna.
        for compost in data["Compost"].unique(): # Devuelve una lista con los valores únicos que aparecen en esa columna.
                                                # Obtiene los valores únicos --> ['Sí', 'No'].
            # Esto filtra el DataFrame 'data' para quedarse solo con las filas que cumplen ambas condiciones.
            # 'subset' es un DataFrame más pequeño que contiene solo los datos de un grupo.
            # Cada subset genera un grupo de puntos con: color según la estación, símbolo según el compost, tamaño según el crecimiento.
            subset = data[(data["Estacion"]==estacion) & (data["Compost"]==compost)]
            ax.scatter( # dibujan en el lienzo un punto específico
                subset["Sol"], subset["Agua"], subset["Crecimiento"],
                c=colores[estacion],
                marker=simbolos[compost],
                s=subset["Crecimiento"],  # tamaño proporcional al crecimiento
                label=f"{estacion}, Compost: {compost}", # 'legend' se diseña a partir de estos elementos
                alpha=0.7
            )

    # se dibuja en el gráfico 3D el punto del nuevo valor predicho <---
    ax.scatter( # dibuja un gráfico de dispersión (scatter plot), es decir, puntos en el espacio (x, y, z).
        nuevo_valor["Sol"][0], # Es el valor del eje X (horas de sol). selecciona la columna “Sol”. toma el primer (y único) valor de esa columna.
        nuevo_valor["Agua"][0], # Accedés a la columna "Agua". [0] toma su único valor.
        prediccion, # valor predicho por el modelo. es el valor Z, que representa el crecimiento estimado de la planta (en milímetros).
                    # este valor se usa como la altura del punto (eje Z).
        c="black",            # color distinto. Define el color del punto.
        marker="X",           # símbolo diferente. Define la forma del punto (símbolo).
                                # formas disponibles: "o": círculo. "^": triángulo. "s": cuadrado. "X": cruz grande (en forma de X)
        s=500,                # tamaño mayor. indica el tamaño del marcador (punto).
        label="Nuevo valor predicho" # Este texto se usa en la leyenda (legend) del gráfico.
    )

    ax.set_xlabel("Sol (horas)")
    ax.set_ylabel("Agua (ml)")
    ax.set_zlabel("Crecimiento (mm)")
    ax.set_title("Crecimiento de Albahaca según Sol, Agua, Estación y Compost")

    # Leyenda sin duplicados
    # handles: son los “manejadores” visuales de los elementos gráficos (los puntos, líneas o marcadores que aparecen en el gráfico).
    # Es decir, los objetos reales que matplotlib usa para dibujar.
    # labels: son los textos que van en la leyenda y etiquetas que describen el gráfico.
    # handles = [objeto1, objeto2, objeto3]
    # labels = ["Primavera, Compost: Sí", "Primavera, Compost: Sí", "Verano, Compost: No"]
    handles, labels = ax.get_legend_handles_labels()
    # zip(labels, handles): junta cada par (label, handle)
    # [
    #  ("Primavera, Compost: Sí", objeto1),
    #  ("Primavera, Compost: Sí", objeto2),
    #  ("Verano, Compost: No", objeto3)
    #]
    # dict(...): convierte esa lista en un diccionario donde las claves son los labels.
    # En un diccionario, no puede haber claves repetidas, así que los duplicados se eliminan automáticamente.
    # unique = {
    #    "Primavera, Compost: Sí": objeto2, --> (solo uno)
    #    "Verano, Compost: No": objeto3
    #}
    unique = dict(zip(labels, handles))
    # unique.values(): los objetos gráficos (los puntos, o “handles”).
    # unique.keys(): los textos (los “labels”).
    # bbox_to_anchor=(x, y): El 1.05 empuja la leyenda un poco hacia la derecha.El 1 indica la parte superior del gráfico.
    # loc='upper left': la esquina superior izquierda de la leyenda se alineará exactamente con el punto (1.05, 1).
    ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    # Internamente, Matplotlib mide el tamaño de cada etiqueta, título o leyenda
    # y reajusta las posiciones, espacios y márgenes de los ejes para evitar que algo quede afuera o encima de otra cosa
    # (evita superposiciones gráficas)
    # para que cada elemento quede bien distribuido visualmente.
    plt.tight_layout()
    plt.show() # abre la ventana con el gráfico final renderizado y ejecuta el programa de visualización con sus opciones.

if __name__ == '__main__': # ejecuta el programa principal: 'main function'
    main()


