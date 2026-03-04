import pandas as pd

df_otono = df[df["Estacion"] == "otoño"]

columna = df_otono["Crecimiento"]

media = columna.mean()
mediana = columna.median()
rango = columna.max() - columna.min()
varianza = columna.var(ddof=1)          # ddof=1: varianza muestral
desviacion_estandar = columna.std(ddof=1)
coef_variacion = desviacion_estandar / media
q1 = columna.quantile(0.25)   # Primer cuartil
q2 = columna.quantile(0.50)   # Segundo cuartil (mediana)
q3 = columna.quantile(0.75)   # Tercer cuartil









