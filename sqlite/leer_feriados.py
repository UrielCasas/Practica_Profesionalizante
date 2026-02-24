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
from   tkinter import ttk, messagebox
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
 
    pass


def marcar_feriado():
        fecha = cal.selection_get()
        evt_id = cal.get_calevents(date=fecha)
        if not evt_id: # si no es feriado ya, lo marcamos como feriado
            cal.calevent_create(fecha, 'Día Feriado', tags="Feriado")
            print( type(fecha) )
            insertar_feriado( fecha.strftime("%Y-%m-%d") )
        #if not evt_id: # ya está marcado como feriado
        #    messagebox.showinfo("info", fecha.ctime() +' NO es feriado')
        #else:
        #    messagebox.showinfo("info", fecha.ctime() +' ES feriado')
        return

def desmarcar_feriado():
        fecha = cal.selection_get()
        evt_id = cal.get_calevents(date=fecha)
        if evt_id: # solo si es feriado hacemos algo
            print(evt_id)
            # cal.calevent_create(fecha, 'Día Feriado', tags="Feriado")
            cal.calevent_remove(date=fecha)
            delete_feriado( fecha.strftime("%Y-%m-%d") )
        return

ttk.Button(root, text="Feriado", command=marcar_feriado).place(y=220, x=80)
ttk.Button(root, text="No Feriado", command=desmarcar_feriado).place(y=220, x=180)


root.mainloop()
