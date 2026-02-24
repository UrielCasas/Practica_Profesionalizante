#-------------------------------------------------------------------------------
# Name:        main
# Purpose:     Leer varios y procesar archivos .csv con datos de uso de la tarjeta 
#              SUBE usando la librerías pandas y numpy para lectura y pre procesamiento 
#              de los mismos.
#              Analizar datos y cálcular regresión logística usando scikit-learn
#              Crear interface gráfica con tkinter 
#              Graficar con matplotlib
#              Datos origen bajados desde:
#              <agregar link>
#              Se contempla sólo el tipo de transporte "COLECTIVO"
#              se podaron los datos de "TREN", "SUBTE" y "LANCHA"
#              
# Author:      Casas Uriel/Fustet Arnaldo 
#              Basado en código por Simón Polizzi
#
# Created:     23/02/2026
# Copyright:   (c) Casas/Fustet  2025/2026
#-------------------------------------------------------------------------------
# importación de Librerías y Módulos a utilizar en el programa
import tkinter as tk
from   tkinter import ttk
from   tkinter import filedialog, messagebox, PhotoImage
import pandas as pd
import time
from   datetime import datetime as dt
from   datetime import date
import datetime
import threading # permite ejecutar tareas en paralelo sin bloquear la interfaz gráfica
import webbrowser # abre enlaces en el navegador web del usuario
import numpy as np
import matplotlib.pyplot as plt
from   sklearn.linear_model import LogisticRegression # quiere decir: del módulo llamadado sklearn.linear_model, importar el objeto llamado: LogisticRegression
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from   sklearn.utils.validation import check_is_fitted # para verificar si el model está entrenado
from sklearn.metrics import log_loss
import os
import sqlite3
from visorpdf import VisorPdf # clase para mostrar y navegar por un pdf
import json
from tkcalendar import Calendar


# ---------------- VARIABLES GLOBALES ----------------
df = None   # Aquí almacenaremos el DataFrame cargado
# crea un objeto modelo de regresión logística
# max_iter=1000 --> número máximo de iteraciones que el algoritmo realizará para convergencia. Sí
# los datos son complejos, puede necesitar más iteraciones para encontrar los coeficientes óptimos
# Convergencia: nos referimos a cuándo el algoritmo “llega a una solución estable”
# durante su proceso iterativo.Sucede cuando los ajustes de los parámetros ya no cambian significativamente
model = None # Modelo de regresión logística
# X_train (features) y y_train (variable objetivo): valores para entrenar el modelo
# X_test y y_test --> probar predicciones del modelo, y comparar predicciones con realidad

X_train             = None
y_train             = None
X_test              = None
y_test              = None
X                   = None
y                   = None
app_dir             = os.path.dirname(__file__)
directorio          = f"{app_dir}/csv/" # nombre del directorio donde están los archivos
                                        # con los datos originales 
informe             = f"{app_dir}/informe.pdf"                                               
manual              = f"{app_dir}/manual.pdf"
config_path         = f"{app_dir}/config.json"
# los datos de estas variables se leen desde archivo json al que con la trayectoria arch_conf
# formato del archivo json: {"test_size":float,"max_iter":int,"random_state":int,"mostrar_previsualizacion":bool}
config              = None    # recibe los datos leides desde arch_conf
# valores por default de claves en archivo config.json
config_default      = {"mostrar_preview" : True, "random_state" : 2100, "max_iter" : 800, "test_size" : 0.1}
feriados            = []
frds                = []

# Datos de información del software
VERSION             = "1.4.4"
AUTORES             = "Casas Uriel - Fustet Arnaldo"
ANIO                = "2025/2026"
LINK_MANUAL         = "https://github.com/UrielCasas/Practica_Profesionalizante/blob/main/docs/manualdeusuario.pdf"
LINK_DESCARGAS      = "https://github.com/UrielCasas/Practica_Profesionalizante"

# ---------------- FUNCIONES ----------------
def importar_csv():
    """
    Docstring for importar_csv
    Importa un archivo CSV con encabezados.
    """
    global df, X, y
   
    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )

    if (not archivo):
        escribir("Importación CSV cancelada.")
        return

    escribir("Importando CSV: " + archivo)

    try:
        #df = pd.read_csv(archivo, header=0)
        df  = pd.read_csv(archivo, sep=';', header=0)
    except Exception as e:
        messagebox.showerror("Error al importar CSV", str(e))
        return

def leer_dir(dir):
    """
    Docstring for leer_dir
    Escanéa el continido del directorio de origen de datos.
    Lee solo el contenido los archivos que tengan extensión .csv
    retorna un DataFrame con todos los datos de los archivos
    """
    global df
    try:
        contenido = os.scandir(dir)
        dfs = list()
        for e in contenido:
            _, extension = os.path.splitext(e.name)

            if extension.lower() != ".csv":
                continue
        
            data = pd.read_csv(f"{dir}/{e.name}")
            dfs.append(data)

        df = pd.concat(dfs, ignore_index=True)
        return True
    except Exception as e:
        messagebox.showerror("Error al importar directorio", str(e))
        return False

def procesar_df():
    try:
        global df
        print(len(df))
        print(df.head())

        df = df[df['TIPO_TRANSPORTE'] != 'TOTAL']
        print(len(df))
        print(df.head())
        df = df[df['TIPO_TRANSPORTE'] == 'COLECTIVO']
        print(len(df))
        print(df.head())

        df = df.groupby(['DIA_TRANSPORTE'], sort=False)['CANT_TRJ'].sum().reset_index()
        print(len(df))
        print(df.head())

        df.rename(columns={'DIA_TRANSPORTE': 'Fecha', 'CANT_TRJ': 'Cantidad'}, inplace=True)

        # 2. Asegurar que la columna sea tipo datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])

        # 3. Guardar el nombre del día de la semana (Lunes, Martes, etc.)
        #df['dia_semana'] = df['Fecha'].dt.day_name()

        # 4. Alternativa: Guardar el número del día (0=Lunes, 6=Domingo)
        df['DiaSemana'] = df['Fecha'].dt.dayofweek

        def es_feriado(fecha):
            global feriados           
            return (fecha in feriados)

        def es_pandemia(fecha):
            pandemia = [ (pd.Timestamp(dt.strptime('2020-03-20', "%Y-%m-%d").date())), 
                        (pd.Timestamp(dt.strptime('2022-03-31', "%Y-%m-%d").date()))]
            return (fecha >= pandemia[0]) and (fecha <= pandemia[1])

        def obtener_estacion(fecha):
            """
            Determina estación del año de fecha
            """
            mes = fecha.month
            dia = fecha.day
            
            if (mes == 12 and dia >= 21) or (mes in [1, 2]) or (mes == 3 and dia <= 20):
                return 'Verano'
            elif (mes == 3 and dia >= 21) or (mes in [4, 5]) or (mes == 6 and dia <= 20):
                return 'Otoño'
            elif (mes == 6 and dia >= 21) or (mes in [7, 8]) or (mes == 9 and dia <= 20):
                return 'Invierno'
            else:
                return 'Primavera'

        def hay_clases(fecha):
            anio = fecha.year # obtiene el año de fecha
            lectivo = {} # crea diccionario vacío
            # agrega elementos al diccionario, uno por cada año 
            # cada elemento contiene 4 int, en dos pares elementos 0 con 1 y 2 con 3
            # los digitos de cada uno de los int representa aaaammdd
            # indicando incio y fin de los períodos lectivos correspondientes al año 
            lectivo[2020]  =[(pd.Timestamp(dt.strptime('20200309', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20200720', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20200731', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20201211', "%Y%m%d").date()))]
            
            lectivo[2021]  =[(pd.Timestamp(dt.strptime('20210301', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20210719', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20210730', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20211217', "%Y%m%d").date()))]
            
            lectivo[2022]  =[(pd.Timestamp(dt.strptime('20220302', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20220718', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20220729', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20221222', "%Y%m%d").date()))]

            lectivo[2023]  =[(pd.Timestamp(dt.strptime('20230227', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20230717', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20230723', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20231222', "%Y%m%d").date()))]

            lectivo[2024]  =[(pd.Timestamp(dt.strptime('20240301', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20240715', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20240726', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20241220', "%Y%m%d").date()))]

            lectivo[2025]  =[(pd.Timestamp(dt.strptime('20250305', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20250721', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20250802', "%Y%m%d").date())),
                            (pd.Timestamp(dt.strptime('20251222', "%Y%m%d").date()))]

            if (lectivo[anio][0] <= fecha <= lectivo[anio][1]) or (lectivo[anio][2] <= fecha <= lectivo[anio][3]):
                return True # si f está en alguno de los rangos que representan los períodos lectivos
                        # devuelve True
        
            return False # devuelve False, no coincidió con ningún período lectivo

        # Usar .isin() sobre la columna 'Fecha' para saber si está en el array feriados
        # esta_en_feriados = df['Fecha'].isin(feriados)
        # Se creó array paralelo a df, esta_en_feriados, contiene True o False según
        # Fecha es feriado o no
        # lo asignamos a un nuevo campo 'Feriado' 
        # df['Feriado'] = esta_en_feriados

        df['Feriado'] = df['Fecha'].apply(es_feriado)

        # (df['col1'] > 5) & (df['col2'] < 10)
        # df['Pandemia'] = np.where( (df['Fecha'] >= pandemia[0]) & (df['Fecha'] <= pandemia[1]) , True, False)

        df['Pandemia'] = df['Fecha'].apply(es_pandemia)

        df['Estacion'] = df['Fecha'].apply(obtener_estacion)

        df['Clases'] = df['Fecha'].apply(hay_clases)

        df = df[['DiaSemana', 'Feriado', 'Estacion', 'Clases', 'Pandemia', 'Cantidad']]

        if chk_preview_var.get():
            mostrar_preview()

        return True
    
    except Exception as e:
        messagebox.showerror("Error procesando datos", str(e))
        return False

def preparar_modelo():
    try:
        global df, X, y
        status_var.set(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas.")

        # variable binaria de éxito
        # -----------------------------
        # convertir ese valor continuo en una variable binaria para usar regresión logística
        # 1 → “éxito” (Cantidad buena) hubo más viajes que la media
        # 0 → “fracaso” (Cantidad baja)
        # median(): metodo que devuelve el valor medio de la columna.
        umbral = df["Cantidad"].median()
        # crea un array de valores booleanos (True/False) comparando cada fila con la mediana. lleva a cabo una serie de comparaciones
        # y realiza un filtro general a partir del 'umbral' calculado
        df["Exito"] = (df["Cantidad"] > umbral).astype(int)   

        # codificación de categorías
        # -----------------------------
        # pd.get_dummies(): transforma las variables cualitativas y categoricas en columnas binarias (0/1)
        # columns=["Estacion"] --> solo codifica esas columnas.
        # 'Invierno': se usa como categoría de referencia (la base con la que se comparan las demás)
        # Si todas las variables Estacion_Otoño, Estacion_Primavera, Estacion_Verano son 0, entonces la estación es Invierno por defecto
        data_encoded = pd.get_dummies(df, columns=["Estacion"], drop_first=True)
        
        # todas las columnas que usaremos como entrada al modelo:
        # Contiene: DiaSemana, Feriado, Estacion_Otoño, Estacion_Verano, Estacion_Primavera, Pandemia
        # elimina esas dos columnas del DataFrame, dejando solo las variables que el modelo puede usar como entrada
        X = data_encoded.drop(columns=["Cantidad", "Exito"])
        # la variable que queremos predecir --> columna "Exito (1 o 0)"
        # cuando entrenamos el modelo con model.fit(X, y), el modelo aprende cómo cada variable afecta la probabilidad de éxito.
        y = data_encoded["Exito"]
        return True
    except Exception as e:
        messagebox.showerror("Error procesando datos", str(e))
        return False

def importar_datos():
    """
    Docstring for importar_datos
    Escanéa el contenido del usando leer_dir() y 
    procesa la información con procesar_df y preparar_modelo()
    """
    global df, X, y, directorio

    if not leer_dir(directorio):
        return

    if not procesar_df():
        return
    
    if not preparar_modelo():
        return

def entrenar(msj=True):
    global X, y, model, X_train, y_train, X_test, y_test, df, config
    if model is None:
        messagebox.showerror("Alerta","Error interno, no hay definido un ningún model")
        return
    
    if df is None:
        messagebox.showerror("Alerta","No hay cargado ningún dataset")
        return

    # entrenamiento del modelo
    # -----------------------------
    # test_size=0.3: 30% de los datos se usan para prueba (test), 70% para entrenar (train).
    # random_state=42: fija la semilla aleatoria para que la división sea reproducible. Cada vez que se ejecute el código,
    # se obtendra la misma división. La función divide los datos aleatoriamente,
    # pero la aleatoriedad puede generar diferentes divisiones cada vez que se ejecuta el código
    # X_train (features) y y_train (variable objetivo): valores para entrenar el modelo
    # X_test y y_test --> probar predicciones del modelo, y comparar predicciones con realidad
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
    
    # fit(): es el método que ajusta el modelo a los datos de entrenamiento
    # calcula los coeficientes β (Betas) para cada variable en 'X_train' que maximizan la
    # probabilidad de predecir correctamente 'y_train'
    # Guarda estos coeficientes y el intercepto en el objeto model para poder hacer predicciones
    # con predict() o probabilidades con predict_proba().
    # X_test: nuevas filas con las mismas columnas que X_train, pero que el modelo no vio durante el entrenamiento
    # y_test: los resultados reales de esas filas de prueba
    # el modelo usa X_test para predecir y luego comparamos con y_test para ver qué tan bien lo hizo
    model.fit(X_train, y_train)

    if msj:
        messagebox.showinfo("Mensaje", f"Modelo Entrenado!")
    return

def verificar():
    global model, X_test, y_test
    
    if model is None:
        return
    
    try:
        check_is_fitted(model)
    except Exception:
        # el modelo no está entrenado, lo entrenamos antes
        entrenar(False)
    
    pd.options.display.float_format = '{:.2f}'.format
    
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
    
    # 4. Get predicted probabilities for the test set
    # log_loss requires probabilities, not the final class predictions
    y_pred_proba = model.predict_proba(X_test)
    # 5. Calculate the log loss
    # The function takes the true labels and the predicted probabilities
    loss = log_loss(y_test, y_pred_proba)

    # muestra el resultado en pantalla con 2 decimales de presición
    messagebox.showinfo("Resultado", f"Exactitud del modelo: {acc:.2f}\nLog Loss: {loss:.4f}")
    # muestra info por consola
    print(f"Log Loss: {loss:.4f}")
    print(f"Exactitud del modelo: {acc:.2f}")
    
    return

def solicitar_datos():
    punto = None
    
    # Crea la ventana para solicitar al usuario datos
    v = tk.Toplevel(root)
    v.title("Ingresar datos punto de prueba")
    v.geometry("365x190")
    v.resizable(False,False)
    v.transient(root) # La hace modal (bloquea la principal)
    v.grab_set()      # Captura eventos (modalidad)
    
    #tk.Button(v, text="Info", command=lambda:messagebox.showinfo("tamaño ventana",f"ancho: {v.winfo_width()} - alto: {v.winfo_height()}"), width=5).pack(pady=10)

    dir = os.path.dirname(__file__)
    img = PhotoImage(file=f"{dir}/info.png").subsample(5,5)
    img_lbl = tk.Label(v, image=img).place(y=20, x=15)
    
    # Creamos Comboboxes (listas desplegables) para la selección de
    # día de la semana y estación del año
    # Define las opciones para la lista de los combobox 
    dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    estaciones = ["Primavera", "Verano", "Otoño", "Invierno"]
    # Configuramos el estado: 'readonly' para no permitir escribir, 'normal' para permitir
    # con set() establecemos un valor predeterminado inicial ej:"Selecciona un día"
    # con values=lista establecemos la lista de valores
    # get() devuelve la opción seleccionada y current() el número de indice de la opción seleccionada
    dias_combo = ttk.Combobox(v, values=dias, width=25)
    dias_combo.config(state='readonly')
    dias_combo.set("Selecciona un día")
    dias_combo.place(y=20, x=115)

    estaciones_combo = ttk.Combobox(v, values=estaciones, width=25)
    estaciones_combo.config(state='readonly')
    estaciones_combo.set("Selecciona una estación")
    estaciones_combo.place(y=60, x=115)

    # check boxes para clases, feriado y pandemia
    # son values booleanos True/False
    chk_clases_var = tk.BooleanVar(value=False)
    chk_feriado_var = tk.BooleanVar(value=False)
    chk_pandemia_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(v, text="Clases", variable=chk_clases_var).place(y=100, x=115)
    ttk.Checkbutton(v, text="Feriado", variable=chk_feriado_var).place(y=100, x=185)
    ttk.Checkbutton(v, text="Pandemia", variable=chk_pandemia_var).place(y=100, x=255)

    def ok():
        nonlocal punto
        if dias_combo.get() not in dias:
            messagebox.showinfo("Alerta","Por favor seleccione un día de la semana")
            return
        
        if estaciones_combo.get() not in estaciones:
            messagebox.showinfo("Alerta","Por favor seleccione una estación del año")
            return

        if model is None:
            messagebox.showinfo("Alerta","Error interno, no hay definido un ningún model")
            return
        
        if df is None:
            messagebox.showinfo("Alerta","No hay cargado ningún dataset")
            return       
       
        punto = pd.DataFrame({
                    "Estacion":  [estaciones_combo.get()],
                    "DiaSemana": [dias_combo.current()],
                    "Clases":    [chk_clases_var.get()], #[ ("1" if chk_clases_var.get() else "0") ],
                    "Feriado":   [chk_feriado_var.get()], #[ ("1" if chk_feriado_var.get() else "0") ],
                    "Pandemia":  [chk_pandemia_var.get()], #[ ("1" if chk_pandemia_var.get() else "0") ]
                })
        
        graficar(punto)

        #messagebox.showinfo("Alerta", estaciones_combo.get()+' - '+dias_combo.get()+' '+str(dias_combo.current()))
        #v.grab_release()
        #v.destroy()
        return
    
    def cancelar():
        nonlocal punto
        punto = None
        v.grab_release()
        v.destroy()
        return

    # para usar ttk.Frame:
    # s = ttk.Style()
    # Create style used by default for all Frames
    # s.configure('Frame1.TFrame', background='#F3F3F3')
    # footer = ttk.Frame(v,style='Frame1.TFrame', borderwidth=0, height=45)

    footer = tk.Frame(v, relief='flat', borderwidth=0, height=45, bg="#F3F3F3")
    footer.pack(side='bottom', fill='x')

    # Botón para aceptar el ingreso
    ttk.Button(footer, text="Aceptar", command=ok).place(y=15, x=10)
    ttk.Button(footer, text="Cancelar", command=cancelar).place(y=15, x=280)


    # Centramos la ventana de carga sobre la principal
    v.update_idletasks()
    posx = root.winfo_x() + root.winfo_width() // 2 - v.winfo_width() // 2
    posy = root.winfo_y() + root.winfo_height() // 2 - v.winfo_height() // 2
    v.geometry(f"+{posx}+{posy}")

    # espera que el usuario cargue los datos
    root.wait_window(v)
    return punto # devuelve el punto si se precionó ok o None si se cerró la ventana

def graficar(nuevo_punto):
    global model

    #nuevo_punto = solicitar_punto()
    #return
    
    if not check():
        return

    # Nuevo punto (de prueba) --> COLOCA EL NUEVO PUNTO Y LO FORMATEA
    # -----------------------------
    # ingresamos nuevo dato al Dataset seteados con los siguientes valores:
    #nuevo_punto = pd.DataFrame({"Estacion": ["Primavera"],"DiaSemana": [4],"Clases": [1],"Feriado": [1],"Pandemia": [0]})
    #nuevo_punto = solicitar_punto()
    if nuevo_punto is None:
        return 
        
    # Codificar igual que entrenamiento
    # pd.get_dummies(...): Convierte variables categóricas (texto) en columnas numéricas
    # binarias (0 o 1), para que el modelo de regresión logística pueda trabajar con ellas
    # drop_first=True: elimina columnas redundates para evitar "multicolinealidad"
    nuevo_punto_encoded = pd.get_dummies(nuevo_punto, columns=["Estacion"], drop_first=True)

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
    # Cuando DiaSemana = 0, Feriado = 0, Clases = 0, Estacion = Invierno (porque es la base) y Pandemia = 0,
    # la tendencia del modelo es fuertemente negativa (muy baja probabilidad de éxito)
    # cada variable multiplica las odds (razón de éxito/fracaso)
    print(f"\nIntercepto del modelo: {model.intercept_[0]:.4f}")

    # curva sigmoide según día de la semana --> COMIENZA A DIBUJAR
    # -----------------------------
    dias_values = np.linspace(0,6,100) # Genera 100 valores equidistantes entre 0 y 6 (día de la semana 0 = Lunes)
    
    # Para graficar solo en función del día de la semana, necesitamos mantener constantes las otras variables: Feriado, Clases, Estación y Pandemia
    # cualquier cambio o variación en la probabilidad se debe solo al día de la semana
    clases_constante   = "0"
    estacion_constante = "Otoño"
    pandemia_constante = "0"
    feriado_constante  = "0"

    # creamos un DataFrame con 100 filas: cada fila es un escenario distinto de 'DiaSemana', pero las demás variables fijas (constantes)
    # se usará para pasar al modelo y obtener la probabilidad de éxito para cada valor de día de la semana.
    df_pred = pd.DataFrame({
        "DiaSemana": dias_values,
        "Feriado": [clases_constante]*100,
        "Clases": [feriado_constante]*100,
        "Estacion": [estacion_constante]*100,
        "Pandemia": [pandemia_constante]*100
    })

    # le pasamos como argumento 'df_pred' --> Dataframe en donde solo varía el valor de día de la semana
    # las columnas 'Estacion' es de tipo texto
    # get_dummies: la convierte en columnas binarias (0/1) según corresponda
    # esto se hace para que el modelo reciba solo números en lugar de texto, ya que el modelo trabaja mejor con valores numéricos
    # o homogeneos. La categoría que se elimina con drop_first=True (en este caso 'Invierno')
    # se asume automáticamente cuando la columna que quedó vale = 0. Originalmente internamente se generan
    # 3 columnas llamadas: 'Estacion_Primavera', 'Estacion_Otoño' y 'Estacion_Verano'. Cuando drop_first=True se elimina la columna 1° --> 'Estacion_Invierno'
    # Estacion_Primavera, Estacion_Otoño y Estacion_Verano son igual a 0 entonces Estacion_Invierno = 1
    df_pred_encoded = pd.get_dummies(df_pred, columns=["Estacion"], drop_first=True)
    #print(df_pred_encoded)
   
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
    # modificado a 6 pul. Alto
    plt.figure(figsize=(8,6))
    # dias_values: eje X: valores del día de la semana 0 = Lun, 1 = Mar,...,6 = Dom
    # prob: eje Y: probabilidad de éxito predicha por el modelo para cada día de la semana
    # color='red': la línea será roja
    # linewidth=2: grosor de la línea = 2 píxeles
    # label='Curva sigmoide': texto para la leyenda del gráfico
    # dibuja la curva en forma de S de la regresión logística, mostrando cómo cambia la probabilidad de éxito según el día de la semana
    plt.plot(dias_values, prob, color='red', linewidth=2, label='Curva sigmoide')

    # configuramos las etiquetas del eje de las X
    # para usar valores mas representativos
    etiquetas_x = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
    valores_x = [0,1,2,3,4,5,6]
    plt.xticks(valores_x, etiquetas_x, rotation=45, ha='right')
   
   
    # dibuja un punto específico en el gráfico, representando nuestro nuevo dato de prueba
    # nuevo_punto["DiaSemana"]: coordenada X: día de la semana del nuevo punto
    # probabilidad: coordenada Y: probabilidad de éxito del modelo para ese punto
    # color='blue': el punto será azul
    # s=100: tamaño del marcador = 100
    # marker='X': forma del marcador = una X grande
    # label='Nuevo punto de prueba': etiqueta para la leyenda
    # Marca en el gráfico dónde cae el nuevo punto según la predicción del modelo
    # aparece sobre la curva si es coherente, o por debajo/arriba según la probabilidad
    plt.scatter(nuevo_punto["DiaSemana"], probabilidad, color='blue', s=100, marker='X', label='Nuevo punto de prueba')
    # Agrega un título al gráfico
    plt.title("Curva Sigmoide - Probabilidad de éxito según día de la semana")
    # Etiqueta para el eje X del gráfico
    plt.xlabel("Día de la semana")
    # Etiqueta para el eje Y del gráfico
    plt.ylabel("Probabilidad de éxito")    

    # Muestra líneas de la cuadrícula para ayudar a leer los valores
    # linestyle='--': líneas punteadas
    # alpha=0.7: transparencia = 70%, para que la cuadrícula no tape la información.
    plt.grid(True, linestyle='--', alpha=0.7)
    # muestra la leyenda del gráfico, usando los label que definimos en plot() y scatter()
    plt.legend()
    
    # para mostrar los datos del eje X en orden decreciente
    # plt.gca().invert_xaxis()
    #plt.show(block=True)
    plt.show()

def check():
    if model is None:
        messagebox.showinfo("Alerta","Error interno, no hay definido ningún modelo")
        return False
    
    if df is None:
        messagebox.showinfo("Alerta","No hay cargado ningún dataset")
        return False
    
    try:
        check_is_fitted(model)
    except Exception:
        # el modelo no está entrenado, lo entrenamos antes
        entrenar()

    return True

def grafico_barras_dias():
    # gráfico de barras: porcentaje de éxito por estación
    # -----------------------------------------------------
    # calcula cuántos éxitos (1) y fracasos (0) hubo por día

    if not check():
        return

    resumen_dias = df.groupby("DiaSemana")["Exito"].mean() * 100  # promedio de éxito por estación (%)

    orden_dias = [0, 1, 2, 3, 4, 5, 6] # ordenar las estaciones según un orden lógico
    # reindex(orden_estaciones) --> sirve para reordenar manualmente las estaciones (o cualquier categoría)
    # en el orden que nosotros queramos mostrar en el gráfico, sin cambiar los valores originales
    resumen_dias = resumen_dias.reindex(orden_dias)
    print('ini')
    print(resumen_dias)
    print('fin')
    # colores distintos para cada barra
    colores = ["#66BB6A", "#FFD54F", "#FF8A65", "#90CAF9", "#FFD54F", "#FF8A65", "#90CAF9"]  # verde, amarillo, naranja, celeste

    # crear gráfico de barras
    plt.figure(figsize=(7,6)) # tamaño total del gráfico. lo que va a ocupar dentro del plano
    barras = plt.bar(resumen_dias.index, resumen_dias.values,  color=colores, edgecolor="black")

    # Agregar etiquetas encima de cada barra
    for barra in barras:
        altura = barra.get_height() # toma el valor de la altura
        # barra.get_x(): Devuelve la posición X de la barra... barra.get_width()/2:Toma la mitad del ancho de la barra. Sirve para centrar el texto sobre la barra
        plt.text(barra.get_x() + barra.get_width()/2, altura + 1, # altura+1:pone el texto centrado en la parte superior de la barra. sumas 1 para que el texto no quede pegado a la barra
                # ha='center': Centra el texto sobre la barra horizontalmente
                # va='bottom': El texto se coloca arriba de la barra, con la parte inferior alineada con altura+1.
                 f"{altura:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Porcentaje de Éxito por Día")  # Título y etiquetas
    plt.xlabel("Día del año")
    plt.ylabel("Porcentaje de éxito (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # configuramos las etiquetas del eje de las X
    # para usar valores mas representativos
    etiquetas_x = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
    valores_x = [0,1,2,3,4,5,6]
    plt.xticks(valores_x, etiquetas_x, rotation=45, ha='right')

    plt.show()

def grafico_barras():
    # gráfico de barras: porcentaje de éxito por estación
    # -----------------------------------------------------
    # calcula cuántos éxitos (1) y fracasos (0) hubo por estación

    if not check():
        return

    resumen_estacion = df.groupby("Estacion")["Exito"].mean() * 100  # promedio de éxito por estación (%)

    orden_estaciones = ["Primavera", "Verano", "Otoño", "Invierno"] # ordenar las estaciones según un orden lógico
    # reindex(orden_estaciones) --> sirve para reordenar manualmente las estaciones (o cualquier categoría)
    # en el orden que nosotros queraos mostrar en el gráfico, sin cambiar los valores originales
    resumen_estacion = resumen_estacion.reindex(orden_estaciones)

    print('ini')
    print(resumen_estacion)
    print('fin')

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

def estadisticas():
    
    if df is None:
        messagebox.showinfo("Alerta","No hay cargado ningún dataset")
        return
    
    columna = df["Cantidad"]

    minimo              = round(columna.min(),2)
    maximo              = round(columna.max(),2)
    media               = round(columna.mean(),2)
    mediana             = round(columna.median(),2)
    rango               = round(maximo - minimo,2)
    varianza            = round(columna.var(ddof=1),2)     # ddof=1: varianza muestral
    desviacion_estandar = round(columna.std(ddof=1),2)
    coef_variacion      = round(desviacion_estandar / media,2)
    q1                  = round(columna.quantile(0.25),2)   # Primer cuartil
    q2                  = round(columna.quantile(0.50),2)   # Segundo cuartil (mediana)
    q3                  = round(columna.quantile(0.75),2)   # Tercer cuartil

    #print(desviacion_estandar)

    resultados = {
        "TIPO"  : [ 
                    "         valor mínimo ",
                    "         valor máximo ",
                    "                media ",
                    "              mediana ",
                    "                rango ",
                    "             varianza ",
                    " desviacion estándard ",
                    "coeficiente variación ",
                    "       primer cuartil ",
                    "      segundo cuartil ",
                    "       tercer cuartil ",],
        "RESULTADO" : [
                    minimo,
                    maximo,
                    media,
                    mediana,
                    rango,
                    varianza,
                    desviacion_estandar,
                    coef_variacion,
                    q1,
                    q2,
                    q3,
        ],
    }
    pd.options.display.float_format = '{:.2f}'.format
    estadisticas = pd.DataFrame(resultados)   
    
    mostrar_datos("Estadísticas", estadisticas)
    return

def mostrar_datos(titulo, datos):
    
    #if datos == None:
    #    return
    
    top = tk.Toplevel()
    top.title(titulo)

    top.geometry("500x400")

    text_area = tk.Text(top, height=20, width=60)
    text_area.pack()

    if isinstance(datos, pd.core.frame.DataFrame):
        text_area.insert(tk.END, datos.to_string()+"\n")
    elif isinstance(datos, pd.core.frame.DataFrame):
        text_area.insert(tk.END, datos)
    else:
        text_area.insert(tk.END, "tipo de datos no contemplado")    
        
    tk.Button(top, text="Cerrar", command=top.destroy).pack(pady=10)

    return

def actualizar_hora():
    """actualiza la hora en la etiqueta cada segundo."""
    hora_actual = time.strftime("%Y-%m-%d %H:%M:%S")
    hora_var.set("Fecha y Hora: " + hora_actual)
    # after(ms, función) ejecuta la función luego de ms milisegundos sin bloquear
    root.after(1000, actualizar_hora)

def escribir(texto):
    """muestra texto en el panel derecho."""
    if texto.strip() == '':
        text_output.delete('1.0', tk.END)
        return
    text_output.insert("end", texto + "\n")
    #text_output.see("end")  # auto-scroll hacia el final

def mostrar_preview():
    """Muestra las primeras filas del DataFrame."""
    text_output.delete("1.0", "end")
    if (df is None):
        escribir("No hay datos cargados.")
        return
    #escribir(df.head(10).to_string())
    #escribir(df.head(100).to_string())
    escribir(df.to_string())   

def cuenta_regresiva_y_salir():
    """cuenta regresiva antes de salir."""
    for i in [3, 2, 1]:
        status_var.set(f"Saliendo en {i} ...")
        escribir(f"Saliendo en {i} ...")
        time.sleep(1)
    root.after(0, root.destroy)

def mostrar_previsualizacion():
    """
    """
    if df is None:
        return
    
    if chk_preview_var.get():
        mostrar_preview()
    else:
        escribir(" ")

def salir():
    """pregunta y, si acepta, hace cuenta regresiva sin trabar la UI."""
    if (messagebox.askyesno("Confirmar", "¿Desea salir?")):
        hilo = threading.Thread(target=cuenta_regresiva_y_salir, daemon=True)
        hilo.start()

def abrir_manual():
    webbrowser.open(LINK_MANUAL)

def mostrar_info():
    # Creamos una ventana con información del programa y autores.
    ventana_info = tk.Toplevel(root, bg="white")        # Ventana "hija" de la principal (root), y el color del fondo será blanco.
    ventana_info.title("Información del Sistema")       # Establece un título a la ventana.
    ventana_info.geometry("350x170")                    # Establece la dimensión de la ventana.

    # Bloque de texto informativo
    info_text = f"Versión: {VERSION}\nAutores: {AUTORES}\nAño: {ANIO}\nDescargas:"        # Se guarda en una variable lo que se mostrara en la ventana "hija".
    tk.Label(ventana_info,                              # Selección donde se ingresara texto que el usuario no podrá editar.
             text=info_text,                            # Insertamos el texto que estaba guardado en la variable 'text'.
             justify="left",                            # Alínea el texto a la izquierda.
             bg="white",                                # El color del fondo será blanco
             pady=10                                    # Agrega rellenos arriba y abajo para que el texto no esté pegado a los bordes.
             ).pack(padx=(0, 135))                      # Es el administrador de geometría, ajusta y muestra el texto en la ventana y lo mueve 135 pixeles a la izquierda.

    # Hipervínculo clickeable
    link_descarga = tk.Label(ventana_info,              # Selección donde se ingresara texto que el usuario no podrá editar.
                             text=f"{LINK_DESCARGAS}",  # Insertamos el texto que estaba guardado en la variable 'LINK_DESCARGAS'.
                             fg="blue",                 # Seleccionamos el color que será el texto.
                             bg="white",                # El color del fondo será blanco
                             cursor="hand2")            # La flecha del mouse para a ser la mano cuando pasa por encima del link.
    link_descarga.pack()                                # Es el administrador de geometría, ajusta y muestra el texto en la ventana.                          

    # Al hacer click, abre la página
    link_descarga.bind("<Button-1>",                                # Al hacer un click, ejecuta la instrucción.
                       lambda e:webbrowser.open(LINK_DESCARGAS))    # Recibe información de .bind() y la recibe lambda e:
    
    # Franja Gris Inferior
    frame_inferior = tk.Frame(ventana_info,             # Selección donde se ingresara texto que el usuario no podrá editar.
                              bg="#F4F4F4")           # El color del fondo será blanco con un pequeño tono gris en la parte inferior.
           
    frame_inferior.pack(side="bottom",                  # Colocamos el frame en la zona inferior (bottom).
                        fill="x")                       # Llenamos en todo el eje x.

    # Botón para cerrar la ventana
    tk.Button(frame_inferior,                               # Selección donde se ingresara texto que el usuario no podrá editar.
              text="Cerrar",                                # Insertamos el texto que estaba guardado en la variable 'text'.
              command=ventana_info.destroy,                 # Cierra y elimina por completo la ventana de la memoria.
              width=10                                      # Define el ancho del botón en caracteres.
              ).pack(pady=10, side="right", padx=(0,20))    # Posiciona el botón debajo del link a la derecha alejado del borde unos 20 píxeles, y 10 píxeles alejados de debajo y arriba.

def configuracion():
    global config
    
    # Crea la ventana para solicitar al usuario datos
    v = tk.Toplevel(root)
    v.title("Ingresar datos de configuración")
    v.geometry("365x180")
    v.resizable(False,False)
    v.transient(root) # La hace modal (bloquea la principal)
    v.grab_set()      # Captura eventos (modalidad)
    
    #tk.Button(v, text="Info", command=lambda:messagebox.showinfo("tamaño ventana",f"ancho: {v.winfo_width()} - alto: {v.winfo_height()}"), width=5).pack(pady=10)

    dir = os.path.dirname(__file__)
    img = PhotoImage(file=f"{dir}/info.png").subsample(5,5)
    img_lbl = tk.Label(v, image=img).place(y=20, x=15)

    def validar_int(S):
        """
        Docstring for validate_numeric
        Verifica que la entrada del tk.Entry sea un int positivo
        :param s: string a validar (usada en tk.Entry)
        """        
        if S.isdigit() or S == "":
            # caracteres permitidos
            return True
        else:
            # s contiene algún caracter extraño
            v.bell()
            return False
                
    def validar_float(S):
        """
        Docstring for validar_float
        
        :param S: string con datos a validar
        """
        if S.strip() == "":
            return True
        try:
            float(S)
            return True
        except ValueError:
            v.bell() 
            return False

    val_int   = v.register(validar_int)
    val_float = v.register(validar_float)

    label_max_iter = tk.Label(v, text="max iter:")
    label_max_iter.place(y=80, x=115)
    entry_max_iter =  tk.Entry(v)
    entry_max_iter.insert(0,str(config["max_iter"]))
    entry_max_iter.configure( validate="key", validatecommand=(val_int,'%S') )
    entry_max_iter.place(y=80, x=200)
    
    label_test_size = tk.Label(v, text="test size:")
    label_test_size.place(y=20, x=115)
    entry_test_size =  tk.Entry(v,validate="key")
    entry_test_size.insert(0,str(config["test_size"]))
    entry_test_size.configure( validate="key", validatecommand=(val_float,'%S') )
    entry_test_size.place(y=20, x=200)

    label_random_state = tk.Label(v, text="random state:")
    label_random_state.place(y=50, x=115)
    entry_random_state =  tk.Entry(v)
    entry_random_state.insert(0,str(config["random_state"]))
    entry_random_state.configure( validate="key", validatecommand=(val_int,'%S') )
    entry_random_state.place(y=50, x=200)


    chk_mostrar_preview_var = tk.BooleanVar(value=config["mostrar_preview"])
    ttk.Checkbutton(v, text="Mostra Previsualización", variable=chk_mostrar_preview_var).place(y=110, x=115)

    def ok():
        guardar = False
        global config, model  

        config["mostrar_preview"]  = chk_mostrar_preview_var.get()
        if config["mostrar_preview"] != chk_preview_var.get():
            guardar = True
            chk_preview_var.set(config["mostrar_preview"])
            if df is not None:
                mostrar_previsualizacion()
        
        # chequear si hubo algún cambio
        if  ( config["random_state"] !=  int(entry_random_state.get()) or \
              config["max_iter"] != int(entry_max_iter.get()) or \
              config["test_size"] != float(entry_test_size.get()) ):
            guardar = True
            # messagebox.showinfo("Datos",f"{entry_test_size.get()} - {entry_random_state.get()} - {entry_max_iter.get()}")
            config["random_state"]             = int(entry_random_state.get())
            config["max_iter"]                 = int(entry_max_iter.get())
            config["test_size"]                = float(entry_test_size.get())                       

            if df is not None: # Si hay DataFrame cargado se vuelve a generar/entrenar el model                

                model = LogisticRegression(max_iter=config["max_iter"]) # Modelo de regresión logística

                preparar_modelo()

                entrenar(False)

                verificar()

        if guardar:
            archivo_config = open(f"{app_dir}/config.json",'w')
            json.dump(config, archivo_config)
            archivo_config.close()

        v.grab_release()
        v.destroy()
        return
    
    def cancelar():
        v.grab_release()
        v.destroy()
        return
    
    footer = tk.Frame(v, relief='flat', borderwidth=0, height=45, bg="#F3F3F3")
    footer.pack(side='bottom', fill='x')

    # Botón para aceptar el ingreso
    ttk.Button(footer, text="Aceptar", command=ok).place(y=15, x=10)
    ttk.Button(footer, text="Cancelar", command=cancelar).place(y=15, x=280)


    # Centramos la ventana de carga sobre la principal
    v.update_idletasks()
    posx = root.winfo_x() + root.winfo_width() // 2 - v.winfo_width() // 2
    posy = root.winfo_y() + root.winfo_height() // 2 - v.winfo_height() // 2
    v.geometry(f"+{posx}+{posy}")

    # espera que el usuario cargue los datos
    root.wait_window(v)
    return #punto # devuelve el punto si se precionó ok o None si se cerró la ventana

def crear_archivo_config():
    global config, config_path, config_default
    try:
        config         = config_default
        archivo_config = open(config_path,'w')
        json.dump(config, archivo_config)
        archivo_config.close()
        return True
    except Exception as e:
        messagebox.showerror(f"{e}")
        return False

def es_int(S):
    """
    Docstring for es_int
    Verifica S sea un int positivo
    :param S: string a validar
    """
    if S.strip() == "":
        return False
    try:
        int(S)
        return True
    except ValueError:
        return False
                
def es_float(S):
    """
    Docstring for es_float
    Verifica que S sea un float válido.
    :param S: string con datos a validar
    """
    if S.strip() == "":
        return False
    try:
        float(S)
        return True
    except ValueError:
        return False

# ---------------- FUNCIÓN PARA LEER ARCHIVO CONFIG ----------------

def leer_config():
    global feriados, frds
    try:
        cnx = sqlite3.connect(f'{app_dir}/config.db')
        cursor = cnx.cursor()
        cursor.execute("SELECT Fecha FROM Feriados")        
        filas = cursor.fetchall()
        cursor.close()
        cnx.close()
        frds = [fila[0] for fila in filas]
        feriados = [pd.Timestamp(x) for x in frds]
        return True
    except Exception as e:
        return False
    

def leer_archivo_config():
    """
    Docstring for leer_archivo_config
    """
    global config_path, config
    try:
       
        archivo_config = open(config_path)
        config         = json.load(archivo_config)
        archivo_config.close()

        if    "max_iter" not in config \
        or "mostrar_preview" not in config \
        or "random_state" not in config \
        or "test_size" not in config:
            if not crear_archivo_config():
                messagebox.showerror("Error Fatal", "Error fatal no se puede ejecutar la aplicación.\nContacte al servicio técnico")
                exit()

            messagebox.showinfo("Mensaje", "Se creo un nuevo .json porque se encontro una anomalía en el contenido.")
                
        if not es_int(str(config["max_iter"])) or not es_int(str(config["random_state"])) \
            or not es_float(str(config["test_size"])) or not isinstance(config["mostrar_preview"], bool):
            if not crear_archivo_config():
                messagebox.showerror("Error Fatal", "Error fatal no se puede ejecutar la aplicación.\nContacte al servicio técnico")
                exit()
            
            messagebox.showinfo("Mensaje", "Se creo un nuevo .json porque se encontro una anomalía en el contenido.")

    except Exception as e:
        if not crear_archivo_config():
            messagebox.showerror("Error Fatal", "Error fatal no se puede ejecutar la aplicación.\nContacte al servicio técnico")
            return False
        print(e)
        messagebox.showinfo("Mensaje", "No se encontró archivo config.json y se creó uno nuevo.")

    return True

def insertar_feriado(fch):
    try:
        cnx = sqlite3.connect(f'{os.path.dirname(__file__)}/config.db')
        cursor = cnx.cursor()
        
        query = "INSERT INTO Feriados (Fecha) VALUES (?)"
        cursor.execute( query, (fch,) ) # si se pasa un solo parámetro usa coma al final,
                                        # sino si se pasa un string toma cada caracter como un elemento 
        cnx.commit()
        
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        if cursor: cursor.close()
        if cnx: cnx.close()
 
    pass

def delete_feriado(fch):
    try:
        cnx = sqlite3.connect(f'{os.path.dirname(__file__)}/config.db')
        cursor = cnx.cursor()
        
        query = "DELETE FROM Feriados WHERE Fecha = ?"
        cursor.execute( query, (fch,) ) # si se pasa un solo parámetro usa coma al final,
                                        # sino si se pasa un string toma cada caracter como un elemento 
        cnx.commit()
        
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        if cursor: cursor.close()
        if cnx: cnx.close()

    return

def leer_feriados():
    v = tk.Tk()
    v.title("Feriados")
    v.geometry("400x400")

    # Si se está en macOS se deber cambiar el tema (theme) para que se muestren los colores correctamente
    # style.theme_names() lista los temas disponibles 
    # style = ttk.Style(v) 
    # style.theme_use('clam') 
    hoy = date.today()
    cal = Calendar(v, selectmode = 'day',
                locale='es_ES', year = hoy.year, month = hoy.month, day = hoy.day)
    cal.pack(pady=20)

    # agregar los feriados como eventos, el array frds se leyó de la base de datos sqlite ./config.db
    # de la tabla Feridados ( ID int, Fecha Text ) -> frds = ["2020-01-01,...,"2025-12-25"] 
    for f in frds:
        e = f.split('-') # e[0] = "aaaa", e[1] = "mm", e[2] = "dd"
        cal.calevent_create(datetime.date( int(e[0]), int(e[1]), int(e[2]) ), 'Día Feriado', tags="Feriado")
    
    # Configurar la apariencia del tag "Feriado"
    cal.tag_config("Feriado", background="red", foreground="yellow")

    def marcar_feriado():
        fecha = cal.selection_get()
        evt_id = cal.get_calevents(date=fecha)
        if not evt_id: # si no es feriado ya, lo marcamos como feriado
            cal.calevent_create(fecha, 'Día Feriado', tags="Feriado")
            insertar_feriado( fecha.strftime("%Y-%m-%d") )
 
        return

    def desmarcar_feriado():
        fecha = cal.selection_get()
        evt_id = cal.get_calevents(date=fecha)
        if evt_id: # solo si es feriado hacemos algo
            cal.calevent_remove(date=fecha)
            delete_feriado( fecha.strftime("%Y-%m-%d") )

        return

    ttk.Button(v, text="Feriado", command=marcar_feriado).place(y=220, x=80)
    ttk.Button(v, text="No Feriado", command=desmarcar_feriado).place(y=220, x=180)

    v.mainloop()
# ---------------- CARGA DE DATOS DE LA APLICACION ------------------

leer_config()

# print(feriados)

if not leer_archivo_config():
    exit()

print(config)

model = LogisticRegression(max_iter=config["max_iter"]) # Modelo de regresión logística



# ---------------- INTERFAZ GRÁFICA ----------------
# llama al constructor de Tkinter que crea la ventana base
# sobre esta ventanase se va a colocar todos los demás widgets y controles
root = tk.Tk()
root.title("Ejemplo de Regresión Logística") # fia un título principal
root.geometry("900x600") # establece el tamaño inicial de la ventana en píxeles. 900 píxeles de ancho y 600 píxeles de alto


# variables asociadas a labels y checkbox
# variables especiales llamadas "Variable Classes" que permiten vincular
# datos directamente con widgets (controles)
# cuando cambian, el control de la interfaz también se actualica automáticamente
hora_var = tk.StringVar() # se usa para mostrar la hora actual en una etiqueta (Label)
status_var = tk.StringVar(value="") # se usa para mostrar mensajes de estado en la barra inferior
chk_preview_var = tk.BooleanVar(value=False) # está asociada al Checkbutton "Mostrar previsualización"

chk_preview_var.set(config["mostrar_preview"])


# tk.Menu: crea una barra de menú
# root: se asocia al Tk() principal para que dicho menú aparezca arriba de la ventana
# menubar: será el contenedor general
menubar = tk.Menu(root)

# tk.Menu(menubar): crea un menú desplegable que estará dentro de la barra menubar
# tearoff=0: desactiva esa línea punteada que permite “arrancar” el menú y moverlo en una ventana separada
menu_hacer = tk.Menu(menubar, tearoff=0)
# add_command: agrega una opción al menú
# label="Importar TXT": texto que verá el usuario
# command=importar_txt: función que se ejecutará al hacer clic
# Las acciones, procesos siempre invocan funciones de Python
# pero sin ningún parámetro
#menu_hacer.add_command(label="Importar TXT", command=importar_txt)
menu_hacer.add_command(label="Importar CSV", command=importar_csv)
# agrega una línea divisoria visual para separar grupos de opciones dentro del menú
menu_hacer.add_separator()
menu_hacer.add_cascade(label="Configuración", command=configuracion)
menu_hacer.add_separator()
menu_hacer.add_cascade(label="Feriados", command=leer_feriados)

menubar.add_cascade(label="Hacer", menu=menu_hacer)

menu_ayuda = tk.Menu(menubar, tearoff=0)
menu_ayuda.add_command(label="Manual PDF", command=abrir_manual)
menu_ayuda.add_command(label="Información", command=mostrar_info)
menubar.add_cascade(label="Ayuda", menu=menu_ayuda)
# config(): sirve para modificar propiedades de un widget directamente
# la propiedad menu de la ventana principal especifica qué menú debe mostrar la aplicación
root.config(menu=menubar)

# Frame superior (hora)
# Frame es un contenedor
# sirve para agrupar y organizar otros widgets y controles
# se usa como si fueran bloques o cajas dentro de la ventana
# ttk.Frame: es la versión "moderna" con estilo visual mejorado de tk
# root: indica dentro de qué widget va a estar contenido este frame
# en este caso, root = la ventana principal
frame_top = ttk.Frame(root)
# pack() es un administrador de geometría
# decide cómo se coloca el widget en la ventana
# fill: indica si el widget debe expandirse para llenar espacio
# "x": se estira horizontalmente. "y": se estira verticalmente. "both": en ambas direcciones
# pady = padding vertical (espacio arriba y abajo). Se mide en píxeles. Sirve para separar este frame de lo que esté encima o debajo
frame_top.pack(fill="x", pady=5)
# Crea una etiqueta dentro del frame frame_top
# La etiqueta no tiene texto fijo. Se actualizará automáticamente con el contenido de hora_var
# side="right": indica en qué lado del frame se va a colocar este label
# "right": alineado a la derecha. Otras opciones: "left" (izq.), "top" (arriba), "bottom" (abajo)
# padx=10: Espacio horizontal (izquierda y derecha). Evita que el label quede pegado al borde
ttk.Label(frame_top, textvariable=hora_var).pack(side="right", padx=10)

# 2. Create a Notebook (tab control)
# The main window 'root' is the parent
notebook = ttk.Notebook(root)
notebook.pack(expand=1, fill="both") # Makes the notebook expand and fill the window

# 3. Create frames for each tab
app_tab     = ttk.Frame(notebook)
manual_tab  = VisorPdf(notebook,abrir=False,cerrar=False)
informe_tab = VisorPdf(notebook,abrir=False,cerrar=False)

# 4. Add the frames as tabs to the notebook
notebook.add(app_tab, text='Aplicación')
notebook.add(manual_tab, text='Manual')
notebook.add(informe_tab, text='Informe')

informe_tab.abrir_pdf(informe)

manual_tab.abrir_pdf(manual)

# Frame principal
frame_main = ttk.Frame(app_tab)
# fill: indica hacia dónde se estira el contenedor si hay espacio disponible
# "x": horizontal. "y": vertical. "both": ambas direcciones
# expand=False: el frame mantiene su tamaño mínimo. expand=True: se le permite ocupar el espacio libre restante
frame_main.pack(fill="both", expand=True)

# cuadro de controles
# ttk.Labelframe: es un Frame (contenedor) igual que ttk.Frame, pero con un borde y un título
# se usa para agrupar visualmente controles relacionados
# se ve como un cuadro con un rótulo arriba
# este Labelframe se coloca dentro del frame grande frame_main
# text="Controles": define el título que aparece en el borde del cuadro
# solo Labelframe puede tener título, Frame no
# padding=10: agrega espacio interno entre el borde del cuadro y sus widgets
frame_controls = ttk.Labelframe(frame_main, text="Controles", padding=10)
frame_controls.pack(side="left", fill="y", padx=10, pady=10)
# ttk.Button(...): crea un botón usando el módulo ttk
# frame_controls: Este es el contenedor padre del botón
# text="Importar TXT": Esto define el texto que se ve en el botón
# command=importar_txt: Este es el evento (events) que se ejecuta cuando el usuario hace click
# importar_txt es el nombre de una función
# .pack(): es el método que posiciona el botón dentro del frame
#ttk.Button(frame_controls, text="Importar TXT", command=importar_txt).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Importar CSV", command=importar_datos).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Calcular estadísticas", command=estadisticas).pack(fill="x", pady=5)
#ttk.Button(frame_controls, text="Salir", command=salir).pack(fill="x", pady=5)

#ttk.Button(frame_controls, text="test", command=entrenar).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Entrenar", command=entrenar).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Verificar", command=verificar).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Graficar", command=solicitar_datos).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Barras", command=grafico_barras).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Barras Días", command=grafico_barras_dias).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Salir", command=salir).pack(fill="x", pady=5)

# anchor="w": significa west (oeste), es decir, alineado a la izquierda
# pady=(10, 5): es el padding vertical externo (espacio arriba y abajo del control)
# 10: espacio de 10 píxeles arriba del label. Separa el label de los botones
# 5: espacio de 5 píxeles abajo del label. Separa el label del listbox
#ttk.Label(frame_controls, text="Separador para TXT:").pack(anchor="w", pady=(10, 5))
# Listbox es un control que muestra una lista de elementos donde el usuario puede seleccionar
# uno varios (si se configura)
# frame_controls: este es el widget padre donde va colocado el Listbox
# height=5: define cuántos ítems se muestran visibles al mismo tiempo
# no define el tamaño en píxeles, sino el número de filas visibles
# exportselection=False: la selección se mantiene fija aunque el usuario interactúe con otros widgets
# es lo que queremos en una aplicación normal
# exportselection=True puede deseleccionar el ítem automáticamente si el usuario hace click en algún otro control
#listbox_sep = tk.Listbox(frame_controls, height=5, exportselection=False)
#listbox_sep.pack(fill="x")
# recorre una lista de strings, donde cada string representa un
# separador posible que se mostrará dentro de la Listbox. conjunto de etiquetas visibles para el usuario
# insert(): agrega elementos (items) a la Listbox. "end":Indica que el ítem se agregará al final de la lista
# listbox_sep.selection_set(0): selecciona automáticamente el ítem en el índice 0 al iniciar. el primer
# item. evita que el Listbox quede sin selección, lo que nos ayudar a evitar errores
# cuando el usuario olvida elegir algo
#for item in ["Espacio", ",", ";", "-", "_"]:
#    listbox_sep.insert("end", item)
#listbox_sep.selection_set(0)

# ttk.Checkbutton(...): es un control de checkbox. Permite activar o desactivar una opción
# booleana (Sí / No, Verdadero / Falso)
# variable=chk_preview_var: asocia el checkbox a una variable de control (BooleanVar())
# chk_preview_var.get() == True: checkbox marcado. chk_preview_var.get() == False: checkbox desmarcado
ttk.Checkbutton(frame_controls, text="Mostrar previsualización", variable=chk_preview_var, command=mostrar_previsualizacion).pack(anchor="w", pady=10)

# panel derecho (text)
# tk.Text es un cuadro de texto multilínea. permite mostrar y/o escribir
# texto extenso, con scroll, varias líneas, etc.
# frame_main: es el frame principal del programa
# wrap="none": Controla cómo se comporta el texto cuando llega al borde del widget.
# "none": No hace salto de línea automático. "word": Salta la línea por palabras completas
# "char": Salta la línea por carácter
# si mostramos un DataFrame (tablas), queremos que las filas no se corten; debe poderse
# hacer scroll horizontal si hace falta. Así las columnas se ven completas.
text_output = tk.Text(frame_main, wrap="none")
# side="right": Indica en qué borde del contenedor se colocará el widget
# "left": Lado izquierdo. "right": Lado derecho. "top": Parte superior. "bottom": Parte inferior
# fill="both": Indica cómo debe crecer el widget dentro del espacio asignado
# "none": No cambia de tamaño. "x": Se estira solo horizontalmente
# "y": Se estira solo verticalmente. "both": Se estira en ambas direcciones
text_output.pack(side="right", fill="both", expand=True)

# barra estado
# crea un Label (una etiqueta de texto) que se usará como barra de estado en la parte
# inferior de la ventana
# root: Contenedor destino. textvariable=status_var: Variable asociada al texto que se muestra
# relief="sunken": Tipo de borde visual. Da aspecto “hundido”
# anchor="w": West --> Alineación del texto hacia la izquierda
ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w").pack(fill="x")

# ---------- ENLACES INFERIORES ----------
import webbrowser # importa el módulo webbrowser, que permite abrir enlaces en por medio de un navegador

def abrir_enlace_manual(event):
     # abre la página web indicada en el navegador predeterminado 
    webbrowser.open(LINK_MANUAL)


ttk.Label(root, text=f"Versión: {VERSION}").pack(side="left", padx=15)

# crea un Label (una etiqueta de texto) que actuará como un enlace clickeable
# root: es la ventana principal donde se mostrará el Label
# text="Descargar libro aquí!" → el texto visible para el usuario
# fg="blue": colorea el texto de azul, simulando un hipervínculo
# cursor="hand2": cambia el puntero del mouse a una mano al pasar sobre el texto
link_descarga = tk.Label(root, text="¡Acceder al Manual desde aquí!", fg="blue", cursor="hand2")
# pack() coloca el Label en la interfaz gráfica
# pady=5 agrega un pequeño espacio vertical arriba y abajo del Label para que no quede pegado a otros elementos
link_descarga.pack(pady=5)
# bind() asocia un evento a una función
# "<Button-1>": significa "clic izquierdo del mouse"
# abrir_descarga: función que se ejecutará al hacer clic, y que abrirá la página web
link_descarga.bind("<Button-1>", abrir_enlace_manual)

# iniciar actualización de hora
actualizar_hora()
# es el loop principal de la interfaz gráfica
# la ventana se vuelve interactiva y se queda “escuchando” eventos
root.mainloop()
