# -*- coding: utf-8 -*-
"""
Syntax:
    python Pre_TNC_WildAnimalAI.py
@author: shpeng
"""

#import tkinter as tk
from tkinter import *
import os
import PIL
from PIL import Image


TARGET_WIDTH = 682
TARGET_HIGHTH = 512

source_path = None
dest_path = None

def resize():
    source = source_path.get()
    target = dest_path.get()

    #select all img file
    if not os.path.exists(source):
        print("Couldn't find srouce path:", source)
        return

    basewidth = 300
    imgage = Image.open('fullsized_image.jpg')
    wpercent = (basewidth / float(imgage.size[0]))
    hsize = int((float(imgage.size[1]) * float(wpercent)))
    imgage = imgage.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    imgage.save('resized_image.jpg')

    #get the fotal number

    #create target folder

    return


def main():
    root =  Tk()
    main_frame = Frame(root)
    main_frame.pack()
    source_path = StringVar()
    dest_path = StringVar()

    path_lf = LabelFrame(main_frame, text="Path")
    path_lf.pack(side=TOP, padx=5, pady=5)
    Label(path_lf, text="Source").grid(row=0, column=0, sticky=W, padx=2, pady=2)
    source_path_entry = Entry(path_lf,textvariable=source_path,width=120)
    source_path_entry.grid(row=0, column=1, sticky=W, padx=2, pady=2)
    source_path.set(os.getcwd())

    Label(path_lf, text="Destination").grid(row=1, column=0, sticky=W, padx=2, pady=2)

    dest_path_entry = Entry(path_lf,textvariable=source_path, width=120)
    dest_path_entry.grid(row=1, column=1, sticky=W, padx=2, pady=2)
    dest_path.set(os.getcwd())

    opt_lf = LabelFrame(main_frame, text="Options")
    opt_lf.pack(side=TOP, padx=5, pady=5, fill=X)
    Label(opt_lf, text="Under construction...").grid(row=0, column=0)

    btn_resize = Button(opt_lf, text="Resize", width=10, command=resize)
    btn_resize.grid(row=1,column=0, sticky=W,padx=2, pady=2)

    btn_exit = Button(opt_lf, text="Exit", width=10, command = root.destroy)
    btn_exit.grid(row=1, column=1, sticky=E,padx=2, pady=2)
    root.mainloop()
    return

if __name__ == '__main__':
    main()