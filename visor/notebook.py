import tkinter as tk
from tkinter import ttk

# 1. Create the main window
root = tk.Tk()
root.title("Tkinter Tab Widget Example")
root.geometry("400x300")

# 2. Create a Notebook (tab control)
# The main window 'root' is the parent
notebook = ttk.Notebook(root)
notebook.pack(expand=1, fill="both") # Makes the notebook expand and fill the window

# 3. Create frames for each tab
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)

# 4. Add the frames as tabs to the notebook
notebook.add(tab1, text='Tab 1')
notebook.add(tab2, text='Tab 2')

# 5. Add widgets to the individual tabs
# Widgets in Tab 1
label1 = ttk.Label(tab1, text="Welcome to Tab 1")
label1.grid(column=0, row=0, padx=20, pady=20) # Use a geometry manager within the frame

button1 = ttk.Button(tab1, text="Click Me in Tab 1")
button1.grid(column=0, row=1, padx=20, pady=5)

# Widgets in Tab 2
label2 = ttk.Label(tab2, text="Content for Tab 2")
label2.grid(column=0, row=0, padx=20, pady=20)

# 6. Run the application loop
root.mainloop()
