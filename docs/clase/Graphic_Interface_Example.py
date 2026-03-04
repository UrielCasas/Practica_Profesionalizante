#-------------------------------------------------------------------------------
# Name:        Graphic_Interface_Example
# Purpose:     Mostrar un pequeño ejemplo de GUI basada 100% en lenguaje
#              Python. Tiene Main, Menú de Opciones, Botones, Checkbox, ListBox
#              Cuadro de Texto, Fecha y Hora, Barra de Estado.
#
# Author:      Simón Polizzi
#
# Created:     10/11/2025
# Copyright:   (c) Simón Polizzi 2025
#-------------------------------------------------------------------------------
# importación de Librerías y Módulos a utilizar en el programa
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import pandas as pd
import time
import threading # permite ejecutar tareas en paralelo sin bloquear la interfaz gráfica
import webbrowser # abre enlaces en el navegador web del usuario

# ---------------- VARIABLES GLOBALES ----------------
df = None   # Aquí almacenaremos el DataFrame cargado

# Datos de información del software
VERSION = "1.0.0"
AUTORES = "Autor1 - Autor2"
ANIO = "2025"
LINK_MANUAL = "https://ejemplo.com/manualdeusuario.pdf"
LINK_DESCARGAS = "https://ejemplo.com/descargas"


# ---------------- FUNCIONES ----------------

def actualizar_hora():
    """actualiza la hora en la etiqueta cada segundo."""
    hora_actual = time.strftime("%Y-%m-%d %H:%M:%S")
    hora_var.set("Fecha y Hora: " + hora_actual)
    # after(ms, función) ejecuta la función luego de ms milisegundos sin bloquear
    root.after(1000, actualizar_hora)


def escribir(texto):
    """muestra texto en el panel derecho."""
    text_output.insert("end", texto + "\n")
    text_output.see("end")  # auto-scroll hacia el final


def obtener_separador():
    """lee el separador seleccionado en el Listbox."""
    seleccion = listbox_sep.curselection()
    if (not seleccion):
        return None

    idx = seleccion[0]
    if (idx == 0):
        return "space"
    elif (idx == 1):
        return ","
    elif (idx == 2):
        return ";"
    elif (idx == 3):
        return "-"
    elif (idx == 4):
        return "_"


def importar_txt():
    """importa un archivo .txt usando el separador elegido en el listbox."""
    global df

    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo TXT",
        filetypes=[("Archivos TXT", "*.txt"), ("Todos los archivos", "*.*")]
    )

    if (not archivo):
        escribir("Importación TXT cancelada.")
        return

    sep = obtener_separador()
    if (sep is None):
        messagebox.showwarning("Separador faltante", "Seleccione un separador en la lista.")
        return

    escribir(f"Importando TXT: {archivo} usando separador '{sep}'")

    try:
        if (sep == "space"):
            df = pd.read_csv(archivo, delim_whitespace=True, header=0)
        else:
            df = pd.read_csv(archivo, sep=sep, header=0)
    except Exception as e:
        messagebox.showerror("Error al importar TXT", str(e))
        return

    status_var.set(f"TXT cargado: {len(df)} filas, {len(df.columns)} columnas.")

    if (chk_preview_var.get()):
        mostrar_preview()


def importar_csv():
    """Importa un archivo CSV con encabezados."""
    global df

    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )

    if (not archivo):
        escribir("Importación CSV cancelada.")
        return

    escribir("Importando CSV: " + archivo)

    try:
        df = pd.read_csv(archivo, header=0)
    except Exception as e:
        messagebox.showerror("Error al importar CSV", str(e))
        return

    status_var.set(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas.")

    if chk_preview_var.get():
        mostrar_preview()


def mostrar_preview():
    """Muestra las primeras filas del DataFrame."""
    text_output.delete("1.0", "end")
    if (df is None):
        escribir("No hay datos cargados.")
        return
    escribir(df.head(10).to_string())


def cuenta_regresiva_y_salir():
    """cuenta regresiva antes de salir."""
    for i in [3, 2, 1]:
        status_var.set(f"Saliendo en {i} ...")
        escribir(f"Saliendo en {i} ...")
        time.sleep(1)
    root.after(0, root.destroy)


def salir():
    """pregunta y, si acepta, hace cuenta regresiva sin trabar la UI."""
    if (messagebox.askyesno("Confirmar", "¿Desea salir?")):
        hilo = threading.Thread(target=cuenta_regresiva_y_salir, daemon=True)
        hilo.start()


def abrir_manual():
    webbrowser.open(LINK_MANUAL)


def mostrar_info():
    messagebox.showinfo(
        "Información",
        f"Versión: {VERSION}\nAutores: {AUTORES}\nAño: {ANIO}\nDescargas: {LINK_DESCARGAS}"
    )


# ---------------- INTERFAZ GRÁFICA ----------------
# llama al constructor de Tkinter que crea la ventana base
# sobre esta ventanase se va a colocar todos los demás widgets y controles
root = tk.Tk()
root.title("Ejemplo de Intefaz Gráfica") # fia un título principal
root.geometry("900x600") # establece el tamaño inicial de la ventana en píxeles. 900 píxeles de ancho y 600 píxeles de alto

# variables asociadas a labels y checkbox
# variables especiales llamadas "Variable Classes" que permiten vincular
# datos directamente con widgets (controles)
# cuando cambian, el control de la interfaz también se actualica automáticamente
hora_var = tk.StringVar() # se usa para mostrar la hora actual en una etiqueta (Label)
status_var = tk.StringVar(value="Listo.") # se usa para mostrar mensajes de estado en la barra inferior
chk_preview_var = tk.BooleanVar(value=True) # está asociada al Checkbutton "Mostrar previsualización"

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
menu_hacer.add_command(label="Importar TXT", command=importar_txt)
menu_hacer.add_command(label="Importar CSV", command=importar_csv)
# agrega una línea divisoria visual para separar grupos de opciones dentro del menú
menu_hacer.add_separator()
menu_hacer.add_command(label="Salir", command=salir)
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

# Frame principal
frame_main = ttk.Frame(root)
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
ttk.Button(frame_controls, text="Importar TXT", command=importar_txt).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Importar CSV", command=importar_csv).pack(fill="x", pady=5)
ttk.Button(frame_controls, text="Salir", command=salir).pack(fill="x", pady=5)
# anchor="w": significa west (oeste), es decir, alineado a la izquierda
# pady=(10, 5): es el padding vertical externo (espacio arriba y abajo del control)
# 10: espacio de 10 píxeles arriba del label. Separa el label de los botones
# 5: espacio de 5 píxeles abajo del label. Separa el label del listbox
ttk.Label(frame_controls, text="Separador para TXT:").pack(anchor="w", pady=(10, 5))
# Listbox es un control que muestra una lista de elementos donde el usuario puede seleccionar
# uno varios (si se configura)
# frame_controls: este es el widget padre donde va colocado el Listbox
# height=5: define cuántos ítems se muestran visibles al mismo tiempo
# no define el tamaño en píxeles, sino el número de filas visibles
# exportselection=False: la selección se mantiene fija aunque el usuario interactúe con otros widgets
# es lo que queremos en una aplicación normal
# exportselection=True puede deseleccionar el ítem automáticamente si el usuario hace click en algún otro control
listbox_sep = tk.Listbox(frame_controls, height=5, exportselection=False)
listbox_sep.pack(fill="x")
# recorre una lista de strings, donde cada string representa un
# separador posible que se mostrará dentro de la Listbox. conjunto de etiquetas visibles para el usuario
# insert(): agrega elementos (items) a la Listbox. "end":Indica que el ítem se agregará al final de la lista
# listbox_sep.selection_set(0): selecciona automáticamente el ítem en el índice 0 al iniciar. el primer
# item. evita que el Listbox quede sin selección, lo que nos ayudar a evitar errores
# cuando el usuario olvida elegir algo
for item in ["Espacio", ",", ";", "-", "_"]:
    listbox_sep.insert("end", item)
listbox_sep.selection_set(0)

# ttk.Checkbutton(...): es un control de checkbox. Permite activar o desactivar una opción
# booleana (Sí / No, Verdadero / Falso)
# variable=chk_preview_var: asocia el checkbox a una variable de control (BooleanVar())
# chk_preview_var.get() == True: checkbox marcado. chk_preview_var.get() == False: checkbox desmarcado
ttk.Checkbutton(frame_controls, text="Mostrar previsualización", variable=chk_preview_var).pack(anchor="w", pady=10)

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

def abrir_descarga(event):
     # abre la página web indicada en el navegador predeterminado 
    webbrowser.open("https://ejemplo.com/manualdeusuario.pdf")

# crea un Label (una etiqueta de texto) que actuará como un enlace clickeable
# root: es la ventana principal donde se mostrará el Label
# text="Descargar libro aquí!" → el texto visible para el usuario
# fg="blue": colorea el texto de azul, simulando un hipervínculo
# cursor="hand2": cambia el puntero del mouse a una mano al pasar sobre el texto
link_descarga = tk.Label(root, text="Descargar libro aquí!", fg="blue", cursor="hand2")
# pack() coloca el Label en la interfaz gráfica
# pady=5 agrega un pequeño espacio vertical arriba y abajo del Label para que no quede pegado a otros elementos
link_descarga.pack(pady=5)
# bind() asocia un evento a una función
# "<Button-1>": significa "clic izquierdo del mouse"
# abrir_descarga: función que se ejecutará al hacer clic, y que abrirá la página web
link_descarga.bind("<Button-1>", abrir_descarga)

def filtro_suave():
    if (df is None):
        messagebox.showwarning("Sin datos", "Primero importe un archivo.")
        return
    
    palabra = filtro_text.get("1.0", "end").strip()
    if (palabra == ""):
        messagebox.showinfo("Vacío", "Escriba una palabra para filtrar.")
        return
    
     # máscara: True donde coincide, False donde no
     # df.astype(str): convierte todo el DataFrame a texto
     # .apply(lambda col: ...): ecorre columna por columna
     # col: es una columna completa. se aplica la función a cada columna
     # col.str.contains(palabra, case=False, na=False): Busca si cada celda de la columna contiene la palabra
     # case=False: ignora mayúsculas/minúsculas. na=False: si una celda es NaN, la trata como: “no coincide”: devuelve False
     # mask: es un DataFrame del mismo tamaño, con: True donde la palabra coincide. False donde NO coincide
    mask = df.astype(str).apply(lambda col: col.str.contains(palabra, case=False, na=False))

    # copia del DF solo para mostrar
    # df.copy(): hace una copia para no modificar tu DataFrame original
    df_filtrado = df.copy().astype(str)

    # reemplazar por blanco cuando no coincide
    # deja el valor original cuando la condición es True
    # reemplaza por “ ” (cadena vacía) cuando la condición es False
    df_filtrado = df_filtrado.where(mask, "")
    
    # mostrar máximo 50 filas para evitar que explote el TextBox
    # "1.0": 1 --> fila 1. 0 --> columna 0. Significa: primer caracter del TextBox
    # end: indica el final del TextBox, sin importar cuántas líneas tenga
    text_output.delete("1.0", "end")
    # inserta texto dentro del TextBox
    # toma solo las primeras 50 filas del DataFrame
    # evita que Tkinter se cuelgue con muchísimas filas
    text_output.insert("end", df_filtrado.head(50).to_string())

# tk.Text: crea un cuadro de texto multilínea (TextBox)
# frame_controls: indica en qué contenedor (frame) se va a colocar
filtro_text = tk.Text(frame_controls, height=1, width=20)
# pack: coloca el TextBox en la interfaz
filtro_text.pack(pady=5)
# cuando el usuario hace clic → se ejecuta la función filtro_suave()
ttk.Button(frame_controls, text="Filtro", command=filtro_suave).pack(pady=5)

# iniciar actualización de hora
actualizar_hora()
# es el loop principal de la interfaz gráfica
# la ventana se vuelve interactiva y se queda “escuchando” eventos
root.mainloop()
