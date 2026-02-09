# pip install tkPDFViewer
import tkinter as tk
from tkinter import ttk
from tkPDFViewer import tkPDFViewer as pdf
import os

# Crear la ventana principal
root = tk.Tk()
root.title("Visor de PDF en Solapa")
root.geometry("800x600")

# Crear el Notebook (contenedor de solapas)
notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True, fill='both')

# Crear un Frame para la solapa
frame_pdf = ttk.Frame(notebook)
notebook.add(frame_pdf, text="Documento PDF")

# --- Incorporar el visor de PDF ---
v1 = pdf.ShowPdf()
v2 = v1.pdf_view(
    frame_pdf,
    pdf_location=f"{os.path.dirname(__file__)}/informe.pdf",
    width=100,
    height=100
)
v2.pack(fill='both', expand=True)

root.mainloop()
