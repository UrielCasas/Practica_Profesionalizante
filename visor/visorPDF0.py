# pip install pymupdf Pillow
import tkinter as tk
from tkinter import ttk
import fitz  # PyMuPDF
from PIL import Image, ImageTk
# from tkPDFViewer import tkPDFViewer as pdf
import os

def mostrar_pdf_en_solapa(notebook, ruta_pdf):
    # Crear un frame para la solapa
    frame_pdf = ttk.Frame(notebook)
    notebook.add(frame_pdf, text="PDF Visualizador")

    # Leer el PDF
    doc = fitz.open(ruta_pdf)
    page = doc.load_page(0)  # Cargar primera página
    pix = page.get_pixmap()
    
    # Convertir a imagen de Pillow
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_tk = ImageTk.PhotoImage(img)

    # Mostrar imagen en label dentro del frame
    label_pdf = ttk.Label(frame_pdf, image=img_tk)
    label_pdf.image = img_tk  # Mantener referencia
    label_pdf.pack()

# Configuración de Tkinter
root = tk.Tk()
root.title("App con PDF en Solapa")
root.geometry("800x600")

# Crear Notebook (contenedor de solapas)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Llamar la función con tu archivo

mostrar_pdf_en_solapa(notebook, f"{os.path.dirname(__file__)}/informe.pdf")

root.mainloop()
