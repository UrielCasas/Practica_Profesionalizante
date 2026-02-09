# pip install pymupdf pillow
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import os

class VisorPdf(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # codificar unicode simbolos
        flecha_izq = chr(129120)
        flecha_der = chr(129122)

        self.doc = None
        self.zoom = 1.0
        self.images = []
        self.pasicion = []

        # ---------- toolbar ----------
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="Abrir", command=self.seleccionar_pdf).pack(side="left", padx=2)
        ttk.Button(toolbar, text="+", command=lambda: self.cambiar_zoom(1.2)).pack(side="left")
        ttk.Button(toolbar, text="-", command=lambda: self.cambiar_zoom(0.8)).pack(side="left")
         
        ttk.Button(toolbar, text=flecha_izq, command=self.pagina_anterior).pack(side="left", padx=5)
        ttk.Button(toolbar, text=flecha_der, command=self.pagina_siguiente).pack(side="left")

        ttk.Label(toolbar, text="Página:").pack(side="left", padx=(10, 2))
        self.pagina = ttk.Entry(toolbar, width=5)
        self.pagina.pack(side="left")
        ttk.Button(toolbar, text="Go", command=self.ir_a_pagina).pack(side="left")

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

    # ---------- PDF logic ----------

    def abrir_pdf(self, path):
        self.doc = fitz.open(path)
        self.zoom = 1.0
        self.pagina_actual = 0
        self.todas_las_paginas()

    def seleccionar_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        self.abrir_pdf(path)

    def todas_las_paginas(self):
        self.canvas.delete("all")
        self.images.clear()
        self.pasicion.clear()

        y = 10
        max_width = 0

        for i, page in enumerate(self.doc):
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat)

            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            photo = ImageTk.PhotoImage(img)

            self.images.append(photo)
            self.pasicion.append(y)

            self.canvas.create_image(10, y, image=photo, anchor="nw")
            y += pix.height + 20
            max_width = max(max_width, pix.width)

        self.canvas.config(scrollregion=(0, 0, max_width + 20, y))
        self.actualizar_nro_pagina()
        self.ir_a_indice(self.pagina_actual)

    # ---------- navigation ----------

    def ir_a_indice(self, page_index):
        if page_index < 0 or page_index >= len(self.pasicion):
            return

        self.pagina_actual = page_index
        y = self.pasicion[page_index]
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
            page = int(self.pagina.get()) - 1
            self.ir_a_indice(page)
        except ValueError:
            pass

    # ---------- zoom ----------

    def cambiar_zoom(self, factor):
        if not self.doc:
            return
        self.zoom *= factor
        self.todas_las_paginas()

    # ---------- helpers ----------

    def actualizar_nro_pagina(self):
        if self.doc:
            self.nro_pagina.config(text=f"{self.pagina_actual + 1} / {len(self.doc)}")
        else:
            self.nro_pagina.config(text="- / -")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

root = tk.Tk()
root.title("Tkinter PDF Viewer")
root.geometry("950x700")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

pdf_tab = VisorPdf(notebook)
notebook.add(pdf_tab, text="PDF Viewer")
pdf_tab.abrir_pdf(f"{os.path.dirname(__file__)}/informe.pdf")
root.mainloop()