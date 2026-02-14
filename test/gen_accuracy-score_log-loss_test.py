#-------------------------------------------------------------------------------
# Name:        gen_accuracy-score_log-loss_test
# Purpose:     Leer varios y procesar archivos .csv con datos de uso de la tarjeta 
#              SUBE usando la librerías pandas y numpy para lectura y pre procesamiento 
#              de los mismos.
#              Usando los siguientes rangos:
#              test_size = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#              random_state = range(100,3100,100) -> random_state = [100,200,300,...,2800,2900,3000]
#              max_iter     = range(100,3100,100) -> max_iter     = [100,200,300,...,2800,2900,3000]
#              genera modelo de regresión con cada uno de los valores de max_iter
#              lo entrana con los valores de random_state y max_iter             
#              calcula el accuracy score y log loss para cada uno
#              guarda los resultados en un archivo ./resultdao.csv
#              Se contempla sólo el tipo de transporte "COLECTIVO"
#              se podaron los datos de "TREN", "SUBTE" y "LANCHA"
#              
# Author:      Casas Uriel/Fustet Arnaldo 
#              
# Created:     13/02/2025
# Copyright:   (c) Casas/Fustet  2025
#-------------------------------------------------------------------------------
# importación de Librerías y Módulos a utilizar en el programa
import pandas as pd
import numpy as np
import time
from   datetime import datetime as dt
from   sklearn.linear_model import LogisticRegression # quiere decir: del módulo llamadado sklearn.linear_model, importar el objeto llamado: LogisticRegression
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from   sklearn.utils.validation import check_is_fitted # para verificar si el model está entrenado
from sklearn.metrics import log_loss
from   tkinter import filedialog, messagebox, PhotoImage
import os

df = None   # Aquí almacenaremos el DataFrame cargado
model = None # Modelo de regresión logística
X_train             = None
y_train             = None
X_test              = None
y_test              = None
X                   = None
y                   = None
app_dir             = os.path.dirname(__file__)
directorio          = f"{app_dir}/csv/"

test_size = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

random_state = range(100,3100,100)

max_iter = range(100,3100,100)

def leer_dir(dir):
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
        
        df = df[df['TIPO_TRANSPORTE'] != 'TOTAL']
        
        df = df[df['TIPO_TRANSPORTE'] == 'COLECTIVO']

        df = df.groupby(['DIA_TRANSPORTE'], sort=False)['CANT_TRJ'].sum().reset_index()
        
        df.rename(columns={'DIA_TRANSPORTE': 'Fecha', 'CANT_TRJ': 'Cantidad'}, inplace=True)
       
        df['Fecha'] = pd.to_datetime(df['Fecha'])

        df['DiaSemana'] = df['Fecha'].dt.dayofweek

        def es_feriado(fecha):
            feriados =  [    '2020-01-01','2020-02-24','2020-02-25','2020-03-23','2020-03-24','2020-04-02','2020-04-10'
                        ,'2020-05-01','2020-05-25','2020-06-15','2020-06-20','2020-07-09','2020-07-10'
                        ,'2020-08-17','2020-10-12','2020-11-23','2020-12-07','2020-12-08','2020-12-25'

                        ,'2021-01-01','2021-02-15','2021-02-16','2021-03-24','2021-04-02'
                        ,'2021-05-01','2021-05-24','2021-05-25','2021-06-20','2021-06-21','2021-07-09','2021-08-16'
                        ,'2021-10-08','2021-10-11','2021-11-20','2021-11-22','2021-12-08','2021-12-25'

                        ,'2022-01-01','2022-02-28','2022-03-01','2022-03-24','2022-04-02','2022-04-15'
                        ,'2022-05-01','2022-05-25','2022-06-17','2022-06-20','2022-07-09','2022-08-15'
                        ,'2022-10-07','2022-10-10','2022-11-20','2022-11-21','2022-12-08','2022-12-09','2022-12-25'

                        ,'2023-01-01','2023-02-20','2023-02-21','2023-03-24','2023-04-02','2023-04-06','2023-04-07'
                        ,'2023-05-01','2023-05-25','2023-06-17','2023-06-20','2023-07-09','2023-08-17'
                        ,'2023-10-12','2023-11-20','2023-12-08','2023-12-25'

                        ,'2024-01-01','2024-02-12','2024-02-13','2024-03-24','2024-04-24','2024-04-28','2024-04-29'
                        ,'2024-05-01','2024-05-25','2024-06-17','2024-06-20','2024-06-21','2024-07-09','2024-08-17'
                        ,'2024-10-11','2024-10-12','2024-11-18','2024-12-08','2024-12-25'

                        ,'2025-01-01','2025-03-03','2025-03-04','2025-03-24','2025-04-02','2025-04-18'
                        ,'2025-05-01','2025-05-02','2025-05-25','2025-06-16','2025-06-20','2025-07-09'
                        ,'2025-08-15','2025-08-17','2025-10-10','2025-11-21','2025-11-24','2025-12-08','2025-12-25'
                    ]

            feriados = [pd.Timestamp(x) for x in feriados]
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
            anio = fecha.year
            lectivo = {}
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
                return True
        
            return False
        
        df['Feriado'] = df['Fecha'].apply(es_feriado)

        df['Pandemia'] = df['Fecha'].apply(es_pandemia)

        df['Estacion'] = df['Fecha'].apply(obtener_estacion)

        df['Clases'] = df['Fecha'].apply(hay_clases)

        df = df[['DiaSemana', 'Feriado', 'Estacion', 'Clases', 'Pandemia', 'Cantidad']]

        return True
    
    except Exception as e:
        messagebox.showerror("Error procesando datos", str(e))
        return False

def preparar_modelo():
    try:
        global df, X, y
        umbral = df["Cantidad"].median()
        df["Exito"] = (df["Cantidad"] > umbral).astype(int) 
        data_encoded = pd.get_dummies(df, columns=["Estacion"], drop_first=True)               
        X = data_encoded.drop(columns=["Cantidad", "Exito"])        
        y = data_encoded["Exito"]
        return True
    except Exception as e:
        messagebox.showerror("Error procesando datos", str(e))
        return False

def entrenar(test_size, random_state, msj=True):
    global X, y, model, X_train, y_train, X_test, y_test, df
    try:
        if model is None:
            messagebox.showerror("Alerta","Error interno, no hay definido un ningún model")
            return False
        
        if df is None:
            messagebox.showerror("Alerta","No hay cargado ningún dataset")
            return False
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model.fit(X_train, y_train)

        if msj:
            messagebox.showinfo("Mensaje", f"Modelo Entrenado!")
        return True
    except Exception as e:
        messagebox.showerror("Error procesando datos", str(e))
        return False

def verificar():
    global model, X_test, y_test
            
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)   
    
    y_pred_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)

    return acc,loss

leer_dir(directorio)
procesar_df()

lineas = ["max iter;random state;test size;accuracy score;log loss"]
for var_max_iter in max_iter:
    model = LogisticRegression( max_iter=var_max_iter )
    for var_random_state in random_state:
        for var_test_size in test_size:
            preparar_modelo()
            entrenar( var_test_size, var_random_state, False )
            acc,loss = verificar()
            print(f"{str(var_max_iter)};{str(var_random_state)};{str(var_test_size)};{str(round(acc,2))};{str(round(loss,2))};")
            lineas.append(f"{str(var_max_iter)};{str(var_random_state)};{str(var_test_size)};{str(round(acc,2))};{str(round(loss,2))};")

resultado = open(f"{app_dir}/resultado.csv", "w")
resultado.write('\n'.join(lineas))
resultado.close()

