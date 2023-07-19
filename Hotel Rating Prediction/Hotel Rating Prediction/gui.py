import csv

from MS2 import TestScript
import pickle
import Test1
import Test2
from sklearn import metrics
import os
import customtkinter as customtkinter
import pandas as pd
from tkinter import filedialog
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest
from tkinter import messagebox
from PIL import Image, ImageTk

ls = []
def openFile():
    filepath1 = filedialog.askopenfilename(initialdir="", title="choose tha test file ",filetypes=[("CSV files", '.csv'), ('Text Docs', '.txt'), ('All types', '.*')])
    lineOne = f"Uploaded successfully"
    allLines = [lineOne]
    messagebox.showinfo("Successfully", "\n".join(allLines))
    if (filepath1):
        global count
        Test1.test1(filepath1)

count=0
def openFile2():
    filepath2 = filedialog.askopenfilename(initialdir="", title="choose tha test file2 ",filetypes=[("CSV files", '.csv'), ('Text Docs', '.txt'), ('All types', '.*')])

    lineOne = f"File Uploaded successfully"
    allLines = [lineOne]
    messagebox.showinfo("Successfully", "\n".join(allLines))
    if (filepath2):
        global count2
        Test2.test2(filepath2)

count2 = 0

PATH = os.path.dirname(os.path.realpath(__file__))

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")

root_tk = customtkinter.CTk()
root_tk.geometry("350x350")
root_tk.title("Hotel Rating Prediction")


def change_mode():
    if switch_2.get() == 1:
        customtkinter.set_appearance_mode("dark")
    else:
        customtkinter.set_appearance_mode("light")
image_size = 20
add_folder_image = ImageTk.PhotoImage(
    Image.open(PATH + "/test_images/add-folder.png").resize((image_size, image_size), Image.ANTIALIAS))
pred_image = ImageTk.PhotoImage(
    Image.open(PATH + "/test_images/analytics.png").resize((image_size, image_size), Image.ANTIALIAS))
exit_image = ImageTk.PhotoImage(
    Image.open(PATH + "/test_images/exit.png").resize((image_size, image_size), Image.ANTIALIAS))

root_tk.grid_rowconfigure(0, weight=1)
root_tk.grid_columnconfigure(0, weight=1, minsize=200)

frame_1 = customtkinter.CTkFrame(master=root_tk, width=260, height=200, corner_radius=15)
frame_1.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

frame_1.grid_columnconfigure(0, weight=1)
frame_1.grid_columnconfigure(1, weight=1)
frame_1.grid_rowconfigure(0, minsize=10)


button_1 = customtkinter.CTkButton(master=frame_1, image=add_folder_image, text="Predict Hotel Rating (Regression) ", width=260, height=40,
                                   compound="right", command=openFile)
button_1.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

button_2 = customtkinter.CTkButton(master=frame_1, image=add_folder_image, text="Predict Hotel Rating (Classification)", width=260, height=40,
                                   compound="right", command=openFile2)
button_2.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")


switch_2 = customtkinter.CTkSwitch(master=frame_1,
                                   text="Dark Mode",
                                   command=change_mode)
switch_2.grid(row=5, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

exit = customtkinter.CTkButton(master=frame_1, image=exit_image, text="Exit", width=190, height=40,
                               compound="right", fg_color="#DD4A48", hover_color="#D35B58",
                               command=root_tk.destroy)
exit.grid(row=6, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

root_tk.mainloop()
