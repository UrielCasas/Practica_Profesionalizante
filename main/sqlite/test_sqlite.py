import sqlite3
import os

app_dir = os.path.dirname(__file__)

feriados = []

with sqlite3.connect(f'{app_dir}/config.db') as cnx:
    cursor = cnx.cursor()

    cursor.execute("SELECT Fecha FROM Feriados")
    
    filas = cursor.fetchall()

    feriados = [fila[0] for fila in filas]
   
    print( feriados )
    
exit()

# The cnx se cierra automáticamente cuando se sale del bloque 'with'
