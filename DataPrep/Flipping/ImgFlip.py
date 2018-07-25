# -*- coding: utf-8 -*-
"""
Syntax:
    python ImgResize.py
@author: Shu Peng
"""

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

import os
import cv2
import glob

# openCv installation
FLIP_VERTICALLY = 0
FLIP_HORIZONTALLY = 1
FLIP_BOTH = -1

root = Tk()
APP_TITLE = "ImgFlip v1.0"

main_frame = None

source_path = StringVar()
destination_path = StringVar()
flip_horizontal = IntVar()
flip_vertical = IntVar()
include_sub_folder = IntVar()


def flip():
    source = source_path.get()
    target = destination_path.get()
    flip_option = None
    try:
        horizon = flip_horizontal.get()
        vertical = flip_vertical.get()
    except ValueError:
        messagebox.showerror("Error", "Failed to read flipping options.")
        return

    if horizon:
        flip_option = FLIP_HORIZONTALLY

    if vertical:
        flip_option = FLIP_VERTICALLY

    if horizon and vertical:
        flip_option = FLIP_BOTH

    if not( horizon or vertical):
        messagebox.showerror("Error", "Please set flipping option.")
        return

    # Select all img file
    if not os.path.exists(source):
        print("Couldn't find source path:", source)
        messagebox.showwarning("File not found", "File doesn't exist.")
        return

    # Recursive
    if include_sub_folder.get():
        for root_dir, dirs, files in os.walk(source):
            for sub_dir in dirs:
                sub_source = os.path.join(root_dir, sub_dir)
                source_dir = os.path.join(root_dir, sub_dir)
                target_dir = os.path.join(target, sub_dir)
                if not os.path.exists(target_dir):
                    os.makedirs((target_dir))
                _flip_jpg_folder(source_dir, target_dir, flip_option)
    else:
        if not os.path.exists(target):
            os.makedirs(target)
        _flip_jpg_folder(source, target, flip_option)
    return


def _flip_jpg_folder(source, target, option):
    if option == FLIP_HORIZONTALLY:
        appendix = '(H)'
    elif option == FLIP_VERTICALLY:
        appendix = '(V)'
    elif option == FLIP_BOTH:
        appendix = '(B)'
    for jpg in glob.glob(source+'\*.jpg', recursive=False):
        source_dir, source_filename = os.path.split(jpg)
        target_filename = os.path.join(target, appendix + source_filename)
        print("loading ", jpg)
        img = cv2.imread(jpg)
        f_img = img.copy()
        f_img = cv2.flip(img, option)
        cv2.imwrite(target_filename, f_img)
    return

def browse_path(btn_name):
    dlg_title = ""
    init_dir = os.getcwd()
    if btn_name == "BrowseSource":
        dlg_title = "Select source folder"
        init_dir = source_path.get()
    elif btn_name =="BrowseDestination":
        dlg_title = "Select destination folder"
        init_dir = destination_path.get()
    filename = filedialog.askdirectory(parent=main_frame, title=dlg_title, initialdir=init_dir)
    if filename:
        filename=filename.replace('/', '\\')
        if btn_name == "BrowseSource":
            source_path.set(filename)
        elif btn_name == "BrowseDestination":
            destination_path.set(filename)
    return


def main():
    main_frame = Frame(root)
    main_frame.master.title(APP_TITLE)
    main_frame.pack()

    path_lf = LabelFrame(main_frame, text="Path")
    path_lf.pack(side=TOP, padx=5, pady=5)
    Label(path_lf, text="Source").grid(row=0, column=0, sticky=W, padx=2, pady=2)
    source_path_entry = Entry(path_lf,textvariable=source_path, width=80)
    source_path_entry.grid(row=0, column=1, sticky=W, padx=2, pady=2)

    btn_browse_source = Button(path_lf, text="Browse ...", command=lambda: browse_path("BrowseSource"))
    btn_browse_source.grid(row=0, column=2, sticky=W, padx=5, pady=2)

    Label(path_lf, text="Destination").grid(row=1, column=0, sticky=W, padx=2, pady=2)

    dest_path_entry = Entry(path_lf,textvariable=destination_path, width=80)
    dest_path_entry.grid(row=1, column=1, sticky=W, padx=2, pady=2)

    btn_browse_destin = Button(path_lf, text="Browse ...", command=lambda: browse_path("BrowseDestination"))
    btn_browse_destin.grid(row=1, column=2, sticky=W, padx=5, pady=2)

    source_path.set(os.getcwd())
    destination_path.set(os.getcwd())

    # Option frame
    opt_lf = LabelFrame(main_frame, text="Options")
    opt_lf.pack(side=TOP, padx=5, pady=5, fill=X)

    flp_frame = Frame(opt_lf)
    flp_frame.pack(side=TOP)

    chk_horizontal = Checkbutton(flp_frame, text="Horizontal flipping(H)", variable=flip_horizontal)
    chk_horizontal.pack(side=LEFT)
    chk_vertical = Checkbutton(flp_frame, text="Vertical flipping(V)", variable=flip_vertical)
    chk_vertical.pack(side=LEFT)

    chk_sub_folder = Checkbutton(opt_lf, text="Including Sub-folders", variable=include_sub_folder)
    #chk_sub_folder.grid(row=2, column=0, columnspan=3, padx=2, pady=2)
    chk_sub_folder.pack(side=LEFT)
    include_sub_folder.set(1)

    # Command frame
    cmd_frame = Frame(main_frame)
    cmd_frame.pack(side=TOP, padx=5, pady=5, fill=X)

    btn_resize = Button(cmd_frame, text="Create", width=10, command=flip)
    btn_resize.pack(side=LEFT, padx=80, pady=2)

    btn_exit = Button(cmd_frame, text="Exit", width=10, command=root.destroy)
    btn_exit.pack(side=RIGHT, padx=80, pady=2)
    root.mainloop()
    return


if __name__ == '__main__':
    main()