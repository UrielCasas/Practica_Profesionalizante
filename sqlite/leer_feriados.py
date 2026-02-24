#-------------------------------------------------------------------------------
# Name:        main
# Purpose:     Lee la tabla Feriados de la base de datos sqlite "config.db".
#              Esta tabla contiene los días feriados de 2020 a 2025 en formato
#              aaaa-mm-dd en la columna Fecha
#              Se crea un componente tkcalendar.Calendar y se agrega un evento
#              por cada uno de los feriados.
#              el componente Calendar se inicia en el de hoy y en español. 
#              
# Author:      Casas Uriel/Fustet Arnaldo 
#              Basado en código por Simón Polizzi
#
# Created:     24/02/2026
# Copyright:   (c) Casas/Fustet  2025/2026
#-------------------------------------------------------------------------------
# importación de Librerías y Módulos a utilizar en el programa
import tkinter as tk
from tkcalendar import Calendar
from datetime import date
import datetime
import sqlite3
import os

app_dir = os.path.dirname(__file__)

# leer tabla Feriados de la base sqlite ./config.db
cnx = sqlite3.connect(f'{app_dir}/config.db')
cursor = cnx.cursor()
cursor.execute("SELECT Fecha FROM Feriados")        
filas = cursor.fetchall()
cursor.close()
cnx.close()
frds = [fila[0] for fila in filas] # se crea array frds -> ["2020-01-01",...,"2025-12-25"]

root = tk.Tk()
root.geometry("400x400")

# Si se está en macOS se deber cambiar el tema (theme) para que se muestren los colores correctamente
# style.theme_names() lista los temas disponibles 
# style = ttk.Style(root) 
# style.theme_use('clam') 
hoy = date.today()
cal = Calendar(root, selectmode = 'day',
               locale='es_ES', year = hoy.year, month = hoy.month, day = hoy.day)
cal.pack(pady=20)

# agregar los feriados como eventos
for f in frds:
    e = f.split('-') # e[0] = "aaaa", e[1] = "mm", e[2] = "dd"
    cal.calevent_create(datetime.date( int(e[0]), int(e[1]), int(e[2]) ), 'Día Feriado', tags="Feriado")

# Configurar la apariencia del tag "Feriado"
cal.tag_config("Feriado", background="red", foreground="yellow")

root.mainloop()
