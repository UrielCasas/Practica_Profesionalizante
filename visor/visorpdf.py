#-------------------------------------------------------------------------------
# Name:        visorpdf
# Purpose:     contiene clase VisorPdf para leer pdf.
#              Cuenta con controles apertura/cierre, zoom y avance de página 
#              
# Author:      Casas Uriel/Fustet Arnaldo 
#
# Created:     09/01/2026
#  
# Copyright:   (c) Casas/Fustet  2026
#
#-------------------------------------------------------------------------------
# importación de Librerías y Módulos a utilizar en el programa

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import fitz  # PyMuPDF

class VisorPdf(ttk.Frame):
    """
    Docstring for VisorPdf

    VisorPdf(parent: Any, abrir: bool = True, zoom: bool = True, avrepag: bool = True, irpag: bool = True) -> VisorPdf

    instalar:
    pip install pymupdf pillow

    uso:
    # main.py
    from visorpdf import VisorPdf
    
    root = tk.Tk()
    root.title("Visor PDF")
    root.geometry("700x400")
    
    pdf_tab = VisorPdf(root)
    pdf_tab.pack()
    pdf_tab.abrir_pdf("ruta al pdf")
   
    root.mainloop()

    """
    def __init__(self, parent, abrir=True, cerrar=True, zoom=True, avrepag=True, irpag=True):
        super().__init__(parent)

        # codificar unicode simbolos
        flecha_izq = chr(129120)
        flecha_der = chr(129122)
        mas_zoom   = chr(10133)
        menos_zoom = chr(10134)

        self.doc = None
        self.zoom = 1.0
        self.imagenes = []
        self.posicion = []

        # ---------- toolbar ----------
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        if abrir:
            ttk.Button(toolbar, text="Abrir", command=self.seleccionar_pdf).pack(side="left", padx=2)

        if cerrar:
            ttk.Button(toolbar, text="Cerrar", command=self.cerrar_pdf).pack(side="left", padx=2)
        
        if zoom:
            ttk.Button(toolbar, text=mas_zoom, command=lambda: self.cambiar_zoom(1.2)).pack(side="left")
            ttk.Button(toolbar, text=menos_zoom, command=lambda: self.cambiar_zoom(0.8)).pack(side="left")
        
        if avrepag: 
            ttk.Button(toolbar, text=flecha_izq, command=self.pagina_anterior).pack(side="left", padx=5)
            ttk.Button(toolbar, text=flecha_der, command=self.pagina_siguiente).pack(side="left")

        ttk.Label(toolbar, text="Página:").pack(side="left", padx=(10, 2))
        self.pagina = ttk.Entry(toolbar, width=5)
        self.pagina.pack(side="left")

        if irpag:
            ttk.Button(toolbar, text="ir", command=self.ir_a_pagina).pack(side="left")

        self.nro_pagina = ttk.Label(toolbar, text="- / -")
        self.nro_pagina.pack(side="right", padx=10)

        # ---------- canvas ----------
        self.canvas = tk.Canvas(self, bg="#777")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.pagina_actual = 0

    # ---------- Manipulación del PDF ----------

    def abrir_pdf(self, path):
        try:
            self.doc = fitz.open(path)
            self.zoom = 1.0
            self.pagina_actual = 0
            self.todas_las_paginas()
        except Exception as e:
            messagebox.showerror("Error",f"Error {str(e)} al intentar abrir el pdf: {path}")
            return

    def seleccionar_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("Archivos Pdf", "*.pdf")])
        if not path:
            return
        self.abrir_pdf(path)

    def cerrar_pdf(self):
        if self.doc is None:
            return        
        if not self.doc.is_closed:
            self.doc.close()
        self.doc = None
        self.canvas.delete("all")
        self.imagenes.clear()
        self.posicion.clear()
    

    def todas_las_paginas(self):
        self.canvas.delete("all")
        self.imagenes.clear()
        self.posicion.clear()

        y = 10
        max_width = 0

        for _, pagina in enumerate(self.doc):
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = pagina.get_pixmap(matrix=mat)

            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            foto = ImageTk.PhotoImage(img)

            self.imagenes.append(foto)
            self.posicion.append(y)

            self.canvas.create_image(10, y, image=foto, anchor="nw")
            y += pix.height + 20
            max_width = max(max_width, pix.width)

        self.canvas.config(scrollregion=(0, 0, max_width + 20, y))
        self.actualizar_nro_pagina()
        self.ir_a_indice(self.pagina_actual)

    # ---------- navegación ----------

    def ir_a_indice(self, indice):
        if indice < 0 or indice >= len(self.posicion):
            return

        self.pagina_actual = indice
        y = self.posicion[indice]
        self.canvas.yview_moveto(y / self.canvas.bbox("all")[3])
        self.actualizar_nro_pagina()

    def pagina_siguiente(self):
        if self.doc and self.pagina_actual < len(self.doc) - 1:
            self.ir_a_indice(self.pagina_actual + 1)

    def pagina_anterior(self):
        if self.doc and self.pagina_actual > 0:
            self.ir_a_indice(self.pagina_actual - 1)

    def ir_a_pagina(self):
        if not self.doc:
            return
        try:
            pagina = int(self.pagina.get()) - 1
            self.ir_a_indice(pagina)
        except ValueError:
            pass

    # ---------- zoom ----------

    def cambiar_zoom(self, factor):
        if not self.doc:
            return
        self.zoom *= factor
        self.todas_las_paginas()

    # ---------- varios ----------

    def actualizar_nro_pagina(self):
        if self.doc:
            self.nro_pagina.config(text=f"{self.pagina_actual + 1} / {len(self.doc)}")
        else:
            self.nro_pagina.config(text="- / -")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")
