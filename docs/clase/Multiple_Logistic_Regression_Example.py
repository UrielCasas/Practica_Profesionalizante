#-------------------------------------------------------------------------------
# Name:        Multiple_Logistic_Regression_Example
# Purpose:     Predecir el crecimiento exitoso o no, de una planta de albahaca
#              según condiciones de sol, agua, estación y compost. El Modelo
#               retornará si a partir de los parámetros seteados para una nueva
#               observación y regitro, el mismo es un caso de "Exito" o "Fracaso".
#                Se considera fracaso si el valor del crecimiento de
#               la planta es menor al valor del Umbral fijado.
#
# Author:      Simón Polizzi
# Created:     02/11/2025
#-------------------------------------------------------------------------------

def main(): # programa base/principal

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression # quiere decir: del módulo llamadado sklearn.linear_model, importar el objeto llamado: LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    np.random.seed(42) # indica el patrón
                        # en el que se van generando los valores. puede seleccionar
                        # más valores que otros dependiendo del 'seed'.
    n = 100 # numero total de muestras del estudio

    # generación en tiempo de ejecución de las muestras para rellenar el Dataset de prubea con el que
    # se va a entrenar el Modelo posteriormente
    # -----------------------------
    data = pd.DataFrame({
        "Sol": np.random.uniform(4, 12, n),
        "Agua": np.random.uniform(200, 1500, n),
        "Estacion": np.random.choice(["Primavera", "Verano", "Otoño", "Invierno"], n),
        "Compost": np.random.choice(["Sí", "No"], n)
    })

    # función de crecimiento
    # -----------------------------
    # función que calcula el crecimiento de cada planta según variables
    # de ambiente y contextuales tratando de simular lo mejor posible el mundo real
    # o las muestras reales que pueden llegar a tomarse en un caso de estudio
    def crecimiento(row):
        base = 50
        sol = 60 * row["Sol"]
        agua = (6 * row["Agua"]) / 10

        if (row["Estacion"] == "Primavera"):
            est = 50
        elif (row["Estacion"] == "Verano"):
            est = 50 if (row["Sol"] <= 7) else -30
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

        # Ruido aleatorio que simula que muchas veces una misma muestra aún bajo las mismas condiciones
        # o muy cercanas en relación  a otra en términos experimentales y empíricos, pueden no
        # relacionarse con los resultados obtenidos o esperados
        ruido = np.random.normal(0, 5)

        return (base + sol + agua + est + comp + ruido)

    # data.apply(funcion, axis=1) --> aplica la función crecimiento fila por fila del DataFrame.
    # axis=1 --> indica que se aplica a lo largo de las filas (recorrido horizontal).
    # Se crea la columna "Crecimiento" en el DataFrame.
    # En pandas, el eje 0 o 1 se refiere a cómo se recorre la tabla.
    # .apply() toma cada fila completa y pasa como 'Series' a la función.
    data["Crecimiento"] = data.apply(crecimiento, axis=1)

    # variable binaria de éxito
    # -----------------------------
    # convertir ese valor continuo en una variable binaria para usar regresión logística
    # 1 → “éxito” (crecimiento bueno)
    # 0 → “fracaso” (crecimiento bajo)
    # median(): metodo que devuelve el valor medio de la columna.
    umbral = data["Crecimiento"].median()
    # crea un array de valores booleanos (True/False) comparando cada fila con la mediana. lleva a cabo una serie de comparaciones
    # y realiza un filtro general a partir del 'umbral' calculado
    data["Exito"] = (data["Crecimiento"] > umbral).astype(int)

    # codificación de categorías
    # -----------------------------
    # pd.get_dummies(): transforma las variables cualitativas y categoricas en columnas binarias (0/1)
    # columns=["Estacion", "Compost"] --> solo codifica esas columnas.
    # 'Invierno ': se usa como categoría de referencia (la base con la que se comparan las demás)
    # Si todas las variables Estacion_Otoño, Estacion_Primavera, Estacion_Verano son 0, entonces la estación es Invierno por defecto
    data_encoded = pd.get_dummies(data, columns=["Estacion", "Compost"], drop_first=True)
    # todas las columnas que usaremos como entrada al modelo:
    # Contiene: Sol, Agua, Estacion_Otoño, Estacion_Verano, Compost_Sí
    # elimina esas dos columnas del DataFrame, dejando solo las variables que el modelo puede usar como entrada
    X = data_encoded.drop(columns=["Crecimiento", "Exito"])
    # la variable que queremos predecir --> columna "Exito (1 o 0)"
    # cuando entrenamos el modelo con model.fit(X, y), el modelo aprende cómo cada variable afecta la probabilidad de éxito.
    y = data_encoded["Exito"]

    # entrenamiento del modelo
    # -----------------------------
    # test_size=0.3: 30% de los datos se usan para prueba (test), 70% para entrenar (train).
    # random_state=42: fija la semilla aleatoria para que la división sea reproducible. Cada vez que se ejecute el código,
    # se obtendra la misma división. La función divide los datos aleatoriamente,
    # pero la aleatoriedad puede generar diferentes divisiones cada vez que se ejecuta el código
    # X_train (features) y y_train (variable objetivo): valores para entrenar el modelo
    # X_test y y_test --> probar predicciones del modelo, y comparar predicciones con realidad
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # crea un objeto modelo de regresión logística
    # max_iter=1000 --> número máximo de iteraciones que el algoritmo realizará para convergencia. Sí
    # los datos son complejos, puede necesitar más iteraciones para encontrar los coeficientes óptimos
    # Convergencia: nos referimos a cuándo el algoritmo “llega a una solución estable”
    # durante su proceso iterativo.Sucede cuando los ajustes de los parámetros ya no cambian significativamente
    model = LogisticRegression(max_iter=1000)
    # fit(): es el método que ajusta el modelo a los datos de entrenamiento
    # calcula los coeficientes β (Betas) para cada variable en 'X_train' que maximizan la
    # probabilidad de predecir correctamente 'y_train'
    # Guarda estos coeficientes y el intercepto en el objeto model para poder hacer predicciones
    # con predict() o probabilidades con predict_proba().
    # X_test: nuevas filas con las mismas columnas que X_train, pero que el modelo no vio durante el entrenamiento
    # y_test: los resultados reales de esas filas de prueba
    # el modelo usa X_test para predecir y luego comparamos con y_test para ver qué tan bien lo hizo
    model.fit(X_train, y_train)

    # Evaluación del modelo
    # -----------------------------
     # las variables de entrada que el modelo no vio durante el entrenamiento
     # y_pred: el resultado predicho por el modelo para cada fila de X_test
     # "Le das al modelo un “examen” (X_test) y él responde con sus predicciones"
     # model.predict(X_test): predice si cada planta tendrá éxito o no
    y_pred = model.predict(X_test) # usa el modelo entrenado para hacer predicciones
    # compara las predicciones (y_pred) con los valores reales (y_test)
    # accuracy_score(y_test, y_pred): compara predicciones con la realidad y calcula porcentaje de aciertos
    acc = accuracy_score(y_test, y_pred)
    # muestra el resultado en pantalla con 2 decimales de presición
    print(f"Exactitud del modelo: {acc:.2f}")




    # Nuevo punto (de prueba) --> COLOCA EL NUEVO PUNTO Y LO FORMATEA
    # -----------------------------
    # ingresamos nuevo dato al Dataset seteados con los siguientes valores:
    nuevo_punto = pd.DataFrame({
        "Sol": [7],
        "Agua": [1000],
        "Estacion": ["Verano"],
        "Compost": ["Sí"]
    })

    # Codificar igual que entrenamiento
    # pd.get_dummies(...): Convierte variables categóricas (texto) en columnas numéricas
    # binarias (0 o 1), para que el modelo de regresión logística pueda trabajar con ellas
    # drop_first=True: elimina columnas redundates para evitar "multicolinealidad"
    nuevo_punto_encoded = pd.get_dummies(nuevo_punto, columns=["Estacion", "Compost"], drop_first=True)

    # el nuevo punto espera exactamente las columnas que tenía 'X' durante el entrenamiento anterior
    # si el nuevo punto no posee algunas de las columnas originales (por ejemplo, 'Estacion_Primavera'
    # porque no está presente en este conjunto),se crea esa columna y se setea un 0
    # el modelo espera recibir exactamente las mismas columnas con el mismo orden que usó al entrenar.
    # nos aseguramos que tenga el nombre y orden las columnas totalmente igual al objeto DataFrame
    for col in X.columns:
        if (col not in nuevo_punto_encoded.columns):
            nuevo_punto_encoded[col] = 0
    # reordena las columnas del nuevo punto para que coincidan exactamente con el orden de X.columns
    # aunque tengas las mismas columnas, si están en distinto orden, el modelo interpretaría los números de manera incorrecta.
    nuevo_punto_encoded = nuevo_punto_encoded[X.columns]

    # predicción del nuevo punto (solo una vez)
    # model.predict(): usa el modelo entrenado para predecir la clase del nuevo dato
    # nuevo_punto_encoded: es el punto que convertido a números (0/1) y tiene todas las columnas que el modelo espera
    # devuelve un array con la predicción: [1]: "exito" o [0]: "fracaso"
    prediccion_clase = model.predict(nuevo_punto_encoded)[0] # [0]: como solo tenemos un dato, tomamos el primer (y único) elemento del array
    # model.predict_proba(): devuelve las probabilidades de pertenencia a cada clase (0 y 1) para cada fila del dataset
    # [0,1]: seleccionamos: fila 0 (nuestro único punto) y columna 1 (probabilidad de éxito)
    # nos dice qué tan seguro está el modelo de que este punto sea un éxito (O NO), como un porcentaje
    probabilidad = model.predict_proba(nuevo_punto_encoded)[0,1]

    print(f"\nNuevo punto de prueba:")
    print(nuevo_punto)
    print(f"Predicción del modelo: {'Éxito' if prediccion_clase==1 else 'Fracaso'}")
    # imprime la probabilidad de éxito del nuevo punto, como un porcentaje redondeado a dos decimales
    print(f"Probabilidad de éxito: {probabilidad*100:.2f}%")

    # ccoeficientes del modelo
    # -----------------------------
    print("\nCoeficientes del modelo (influencia de cada variable):")
    # X.columns: contiene los nombres de las variables
    # model.coef_: es una matriz con los coeficientes β₁, β₂, … que el modelo aprendió durante el entrenamiento
    # en regresión logística binaria, model.coef_ tiene forma (1, n_features), por eso se accede con [0] para obtener el vector plano
    # zip(...): empareja cada nombre de variable con su coeficiente correspondiente
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature:25s} -> {coef:.4f}")

    # el intercepto es la 'log-odds base', es decir, el “punto de partida” del modelo cuando todas las variables valen 0
    # Cuando Sol = 0, Agua = 0, Estacion = Invierno (porque es la base) y Compost = No,
    # la tendencia del modelo es fuertemente negativa (muy baja probabilidad de éxito)
    # cada variable multiplica las odds (razón de éxito/fracaso)
    print(f"\nIntercepto del modelo: {model.intercept_[0]:.4f}")


    # curva sigmoide según horas de sol --> COMIENZA A DIBUJAR
    # -----------------------------
    sol_values = np.linspace(4,12,100) # Genera 100 valores equidistantes entre 4 y 12 (horas de sol)
    # Para graficar solo en función del 'sol', necesitamos mantener constantes las otras variables: agua, estación y compost
    # cualquier cambio o variación en la probabilidad se debe solo a la cantidad de sol
    agua_media = data["Agua"].mean()
    estacion_constante = "Otoño"
    compost_constante = "Sí"

    # creamos un DataFrame con 100 filas: cada fila es un escenario distinto de 'sol', pero las demás variables fijas (constantes)
    # se usará para pasar al modelo y obtener la probabilidad de éxito para cada valor de sol.
    df_pred = pd.DataFrame({
        "Sol": sol_values,
        "Agua": [agua_media]*100,
        "Estacion": [estacion_constante]*100,
        "Compost": [compost_constante]*100
    })

    # le pasamos como argumento 'df_pred' --> Dataframe en donde solo varía el valor de hs. totales de 'sol'
    # las columnas 'Estacion' y 'Compost' son tipo texto
    # get_dummies: las convierte en columnas binarias (0/1) según corresponda
    # esto se hace para que el modelo reciba solo números en lugar de texto, ya que el modelo trabaja mejor con valores númericos
    # o homogeneos. La categoría que se elimina con drop_first=True (en este caso 'No')
    # se asume automáticamente cuando la columna que quedó vale = 0. Originalmente internamente se generan
    # 2 columnas llamadas: 'Compost_Sí' y 'Compost_No'. Cuando drop_first=True se elimina la columna 2° --> 'Compost_No'
    # esto hace que el modelo deduzca que 'Sí' = 1 y 'No' = 0.
    # "Estacion_Otoño" = 1 si es otoño, 0 si no.
    df_pred_encoded = pd.get_dummies(df_pred, columns=["Estacion","Compost"], drop_first=True)

    # el modelo entrenado espera exactamente las columnas que tenía 'X' durante el entrenamiento anterior
    # si el nuevo DataFrame de predicción no tiene alguna columna (por ejemplo, 'Estacion_Primavera'
    # porque no está presente en este conjunto), el modelo tira 'error'
    for col in X.columns:
        if (col not in df_pred_encoded.columns):
            df_pred_encoded[col] = 0
    # asegura que el orden de las columnas sea el mismo que el del entrenamiento
    # paso importante ya que los modelos de 'scikit-learn' usan el orden de las columnas para asociarlas con los coeficientes
    df_pred_encoded = df_pred_encoded[X.columns]

    # devuelve un array 2D con una fila por cada ejemplo y dos columnas, una para cada clase:
    # Columna 0 --> probabilidad de que la clase sea 0 (Fracaso)
    # Columna 1 --> probabilidad de que la clase sea 1 (Éxito)
    prob = model.predict_proba(df_pred_encoded)[:,1] # [:,1]: toma todas las filas de la columna 1°:
                                                    # estamos seleccionando la probabilidad de que sea éxito: (1).
                                                    # # [0.76, 0.40, 0.90] --> es un vector 1D con la probabilidad de éxito para cada fila de entrada.

    # gráfico final
    # -----------------------------
    # 1cm. = 2.54 pul. (in)
    # Crea una nueva figura para dibujar el gráfico. figsize=(8,5): define el tamaño del gráfico en pulgadas
    # 8 pul. = Ancho | 5 pul. = Alto --> 20,32cm.X12,7cm.
    plt.figure(figsize=(8,5))
    # sol_values: eje X: valores de horas de sol (4 a 12 hs)
    # prob: eje Y: probabilidad de éxito predicha por el modelo para cada valor de sol
    # color='red': la línea será roja
    # linewidth=2: grosor de la línea = 2 píxeles
    # label='Curva sigmoide': texto para la leyenda del gráfico
    # dibuja la curva en forma de S de la regresión logística, mostrando cómo cambia la probabilidad de éxito según las horas de sol
    plt.plot(sol_values, prob, color='red', linewidth=2, label='Curva sigmoide')
    # dibuja un punto específico en el gráfico, representando nuestro nuevo dato de prueba
    # nuevo_punto["Sol"]: coordenada X: horas de sol del nuevo punto
    # probabilidad: coordenada Y: probabilidad de éxito del modelo para ese punto
    # color='blue': el punto será azul
    # s=100: tamaño del marcador = 100
    # marker='X': forma del marcador = una X grande
    # label='Nuevo punto de prueba': etiqueta para la leyenda
    # Marca en el gráfico dónde cae el nuevo punto según la predicción del modelo
    # aparece sobre la curva si es coherente, o por debajo/arriba según la probabilidad
    plt.scatter(nuevo_punto["Sol"], probabilidad, color='blue', s=100, marker='X', label='Nuevo punto de prueba')
    # Agrega un título al gráfico
    plt.title("Curva Sigmoide - Probabilidad de éxito según horas de sol")
    # Etiqueta para el eje X del gráfico
    plt.xlabel("Horas de sol por día")
    # Etiqueta para el eje Y del gráfico
    plt.ylabel("Probabilidad de éxito")
    # Muestra líneas de la cuadrícula para ayudar a leer los valores
    # linestyle='--': líneas punteadas
    # alpha=0.7: transparencia = 70%, para que la cuadrícula no tape la información.
    plt.grid(True, linestyle='--', alpha=0.7)
    # muestra la leyenda del gráfico, usando los label que definimos en plot() y scatter()
    plt.legend()
    # Debe ser la última línea para visualizar la figura con todo lo que definimos: línea, punto, etiquetas, cuadrícula y leyenda
    # muestra el gráfico final en pantalla.
    plt.show()

     # gráfico de barras: porcentaje de éxito por estación
    # -----------------------------------------------------
    # calcula cuántos éxitos (1) y fracasos (0) hubo por estación
    resumen_estacion = data.groupby("Estacion")["Exito"].mean() * 100  # promedio de éxito por estación (%)

    orden_estaciones = ["Primavera", "Verano", "Otoño", "Invierno"] # ordenar las estaciones según un orden lógico
    # reindex(orden_estaciones) --> sirve para reordenar manualmente las estaciones (o cualquier categoría)
    # en el orden que nosotros queraos mostrar en el gráfico, sin cambiar los valores originales
    resumen_estacion = resumen_estacion.reindex(orden_estaciones)

    # colores distintos para cada barra
    colores = ["#66BB6A", "#FFD54F", "#FF8A65", "#90CAF9"]  # verde, amarillo, naranja, celeste

    # crear gráfico de barras
    plt.figure(figsize=(7,5)) # tamaño total del gráfico. lo que va a ocupar dentro del plano
    barras = plt.bar(resumen_estacion.index, resumen_estacion.values,  color=colores, edgecolor="black")

    # Agregar etiquetas encima de cada barra
    for barra in barras:
        altura = barra.get_height() # toma el valor de la altura
        # barra.get_x(): Devuelve la posición X de la barra... barra.get_width()/2:Toma la mitad del ancho de la barra. Sirve para centrar el texto sobre la barra
        plt.text(barra.get_x() + barra.get_width()/2, altura + 1, # altura+1:pone el texto centrado en la parte superior de la barra. sumas 1 para que el texto no quede pegado a la barra
                # ha='center': Centra el texto sobre la barra horizontalmente
                # va='bottom': El texto se coloca arriba de la barra, con la parte inferior alineada con altura+1.
                 f"{altura:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Porcentaje de Éxito por Estación")  # Título y etiquetas
    plt.xlabel("Estación del año")
    plt.ylabel("Porcentaje de éxito (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()

if __name__ == '__main__': # hace un Call al cuerpo del main/programa principal
    main()





