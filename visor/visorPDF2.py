import tkinter as tk
from tkinter import ttk
import fitz  # PyMuPDF
from PIL import Image, ImageTk
import os

class PDFScrollViewer:
    def __init__(self, notebook, ruta_pdf):
        self.doc = fitz.open(ruta_pdf)
        self.total_paginas = len(self.doc)

        # Frame de la solapa
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="PDF (Scroll continuo)")

        # ----- Barra superior -----
        top = ttk.Frame(self.frame)
        top.pack(fill="x")

        self.lbl_pagina = ttk.Label(top, text="Página 1 / {}".format(self.total_paginas))
        self.lbl_pagina.pack(padx=10, pady=5)

        # ----- Canvas + Scroll -----
        self.canvas = tk.Canvas(self.frame, bg="#ddd")
        self.scroll = ttk.Scrollbar(
            self.frame, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Cargar todas las páginas
        self.page_positions = []
        self._cargar_paginas()

        # Actualizar página visible al hacer scroll
        self.canvas.bind("<Configure>", lambda e: self._update_pagina_visible())
        self.canvas.bind("<Motion>", lambda e: self._update_pagina_visible())
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._update_pagina_visible())

    # ---------------- PDF ----------------
    def _cargar_paginas(self):
        for i in range(self.total_paginas):
            page = self.doc.load_page(i)
            pix = page.get_pixmap()

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            img_tk = ImageTk.PhotoImage(img)

            lbl = ttk.Label(self.inner, image=img_tk)
            lbl.image = img_tk
            lbl.pack(pady=10)

            self.inner.update_idletasks()
            self.page_positions.append(lbl.winfo_y())

    # ---------------- Eventos ----------------
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._update_pagina_visible()

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _update_pagina_visible(self):
        y_scroll = self.canvas.canvasy(0)

        for i, pos in enumerate(self.page_positions):
            if y_scroll < pos + 20:
                self.lbl_pagina.config(
                    text=f"Página {i + 1} / {self.total_paginas}"
                )
                break


# ---------------- Tkinter ----------------
root = tk.Tk()
root.title("Visor PDF - Scroll continuo")
root.geometry("900x700")

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

PDFScrollViewer(notebook, f"{os.path.dirname(__file__)}/informe.pdf")

root.mainloop()
