import tkinter as tk
from tkinter import ttk
import fitz  # PyMuPDF
from PIL import Image, ImageTk
import os


class PDFViewer:
    def __init__(self, notebook, ruta_pdf):
        self.doc = fitz.open(ruta_pdf)
        self.total_paginas = len(self.doc)
        self.pagina_actual = 0

        # Frame de la solapa
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="PDF Visualizador")

        # ----- Canvas + Scroll -----
        self.canvas = tk.Canvas(self.frame)
        self.scroll_y = ttk.Scrollbar(
            self.frame, orient="vertical", command=self.canvas.yview
        )

        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.scroll_y.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Label donde se muestra la imagen
        self.label_pdf = ttk.Label(self.inner_frame)
        self.label_pdf.pack(pady=10)

        # ----- Botones -----
        botones = ttk.Frame(self.frame)
        botones.pack(fill="x")

        ttk.Button(botones, text="⏮ Anterior", command=self.pagina_anterior).pack(
            side="left", padx=5, pady=5
        )
        ttk.Button(botones, text="Siguiente ⏭", command=self.pagina_siguiente).pack(
            side="right", padx=5, pady=5
        )

        self.mostrar_pagina()

    def mostrar_pagina(self):
        page = self.doc.load_page(self.pagina_actual)
        pix = page.get_pixmap()

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        img_tk = ImageTk.PhotoImage(img)
        self.label_pdf.configure(image=img_tk)
        self.label_pdf.image = img_tk  # evitar garbage collector

    def pagina_siguiente(self):
        if self.pagina_actual < self.total_paginas - 1:
            self.pagina_actual += 1
            self.mostrar_pagina()

    def pagina_anterior(self):
        if self.pagina_actual > 0:
            self.pagina_actual -= 1
            self.mostrar_pagina()


# ---------------- Tkinter ----------------
root = tk.Tk()
root.title("App con PDF en Solapa")
root.geometry("900x700")

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Crear visor
# 👉 cambia la ruta a tu PDF
PDFViewer(notebook, f"{os.path.dirname(__file__)}/informe.pdf")

root.mainloop()
