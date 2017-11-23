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
TARGET_WIDTH = 682
TARGET_HEIGHT = 512

root = Tk()
APP_TITLE = "ImgResize v1.2"

main_frame = None

source_path = StringVar()
dest_path = StringVar()

pix_width = StringVar()
pix_height = StringVar()

include_sub_folder = IntVar()


def _resize_jpg(source_path, destin_path, width, height):
    jpg_file_list = glob.glob(os.path.join(source_path,"*.jpg"))

    if not os.path.exists(destin_path):
        print("Creating the target path:", destin_path)
        os.mkdir(destin_path)

    for img_file_name in jpg_file_list:
        # Load an color image
        print("Loading original image file -", img_file_name)
        img = cv2.imread(img_file_name)
        re_img = cv2.resize(img, (width, height))
        # save
        resized_name = os.path.join(destin_path, os.path.basename(img_file_name))
        print("Saving resized image as ", resized_name)
        cv2.imwrite(resized_name, re_img)
    print("Done")
    return


def resize():
    source = source_path.get()
    target = dest_path.get()
    width = TARGET_WIDTH
    height = TARGET_HEIGHT
    try:
        width = int(pix_width.get())
        height = int(pix_height.get())
    except ValueError:
        messagebox.showerror("Error", "Width or height value is wrong")
        return

    # Select all img file
    if not os.path.exists(source):
        print("Couldn't find source path:", source)
        messagebox.showwarning("File not found", "File doesn't exist.")
        return

    if source == target:
        answer = messagebox.askyesno("Warning","Source path and target path are the same, original file(s) will be "
                                             "overwritten, do you still want to continue?" )
        if answer != True:
            return

    if not os.path.exists(target):
        os.mkdir(target)

    # Recursive
    if include_sub_folder.get():
        len_source_path = len(source)
        for root, dirs, files in os.walk(source):
            for sub_dir in dirs:
                sub_source = os.path.join(root, sub_dir)
                sub_target = target + sub_source[len_source_path:]
                _resize_jpg(sub_source, sub_target, width, height)
    else:
        _resize_jpg(source,target, width, height)
    return


def browse_path(btn_name):
    dlg_title = ""
    init_dir = os.getcwd()
    if btn_name == "BrowseSource":
        dlg_title = "Select source folder"
        init_dir = source_path.get()
    elif btn_name =="BrowseDestination":
        dlg_title = "Select destination folder"
        init_dir = dest_path.get()
    filename = filedialog.askdirectory(parent=main_frame, title=dlg_title, initialdir=init_dir)
    if filename:
        filename=filename.replace('/', '\\')
        if btn_name == "BrowseSource":
            source_path.set(filename)
        elif btn_name == "BrowseDestination":
            dest_path.set(filename)
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
    btn_browse_source.grid(row=0, column=2, sticky=W, padx=2, pady=2)

    Label(path_lf, text="Destination").grid(row=1, column=0, sticky=W, padx=2, pady=2)

    dest_path_entry = Entry(path_lf,textvariable=dest_path, width=80)
    dest_path_entry.grid(row=1, column=1, sticky=W, padx=2, pady=2)

    btn_browse_destin = Button(path_lf, text="Browse ...", command=lambda: browse_path("BrowseDestination"))
    btn_browse_destin.grid(row=1, column=2, sticky=W, padx=2, pady=2)

    source_path.set(os.getcwd())
    dest_path.set(os.getcwd())

    # Option frame
    opt_lf = LabelFrame(main_frame, text="Options")
    opt_lf.pack(side=TOP, padx=5, pady=5, fill=X)

    Label(opt_lf, text="Width=").grid(row=0, column=0, sticky=E, padx=2, pady=2)
    width_entry = Entry(opt_lf, textvariable=pix_width, width=10)
    width_entry.grid(row=0, column=1, sticky=W, padx=2, pady=2)
    pix_width.set(TARGET_WIDTH)
    Label(opt_lf, text="pix").grid(row=0, column=2, sticky=W)
    Label(opt_lf, text="Height=").grid(row=1, column=0, sticky=E, padx=2, pady=2)
    width_entry = Entry(opt_lf, textvariable=pix_height, width=10)
    width_entry.grid(row=1, column=1, sticky=W, padx=2, pady=2)
    pix_height.set(TARGET_HEIGHT)
    Label(opt_lf, text="pix").grid(row=1, column=2, sticky=W)

    chk_sub_folder = Checkbutton(opt_lf, text="Including Sub-folders", variable=include_sub_folder)
    chk_sub_folder.grid(row=2, column=0, columnspan=3, padx=2, pady=2)
    include_sub_folder.set(1)

    # Command frame
    cmd_frame = Frame(main_frame)
    cmd_frame.pack(side=TOP, padx=5, pady=5, fill=X)

    btn_resize = Button(cmd_frame, text="Resize", width=10, command=resize)
    btn_resize.pack(side=LEFT, padx=80, pady=2)

    btn_exit = Button(cmd_frame, text="Exit", width=10, command = root.destroy)
    btn_exit.pack(side=RIGHT, padx=80, pady=2)
    root.mainloop()
    return


if __name__ == '__main__':
    main()