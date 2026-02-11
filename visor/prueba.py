#-------------------------------------------------------------------------------
# Name:        prueba.py
# Purpose:     Prueba de la clase VisorPdf
#              
# Author:      Casas Uriel/Fustet Arnaldo 
#
# Created:     09/01/2026
# Copyright:   (c) Casas/Fustet  2025
#-------------------------------------------------------------------------------
# importación de Librerías y Módulos a utilizar en el programa
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from visorpdf import VisorPdf

root = tk.Tk()
root.title("Tkinter PDF Viewer")
root.geometry("950x700")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

pdf_tab = VisorPdf(notebook)
notebook.add(pdf_tab, text="Visor PDF")
pdf_tab.abrir_pdf(f"{os.path.dirname(__file__)}/TP04.pdf")
root.mainloop()