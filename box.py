# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:10:30 2019

@author: HP
"""


import tkinter
from tkinter import messagebox

# hide main window
root = tkinter.Tk()
root.withdraw()

# message box display
messagebox.showerror("Error", "Error message")
messagebox.showwarning("Warning","Warning message")
messagebox.showinfo("Information","Informative message")