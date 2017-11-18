# -*- coding: utf-8 -*-
"""
Syntax:
    python ImgResize.py
@author: Shu Peng
"""

from tkinter import *
from tkinter import messagebox
import os
import PIL
from PIL import Image
import cv2
import glob
# openCv installation
TARGET_WIDTH = 682
TARGET_HIGHTH = 512

root = Tk()
root.title = "ImgResize v0.1"
source_path = StringVar()
dest_path = StringVar()

def resize():
    source = source_path.get()
    target = dest_path.get()

    #select all img file
    print("Target Folder: ",target)
    if not os.path.exists(source):
        print("Couldn't find source path:", source)
        messagebox.showwarning("File not found", "File doesn't exist.")
        return

    if source == target:
        aswr = messagebox.askyesno("Warning","Source path and target path are the same, original file(s) will be "
                                             "overwritten, do you still want to continue?" )
        if aswr != True:
            return

    jpg_file_list = glob.glob(os.path.join(source,"*.jpg"))

    if not os.path.exists(target):
        print("Creating the target path:", target)
        os.mkdir(target)

    for img_file_name in jpg_file_list:
        # Load an color image in grayscale
        img = cv2.imread(img_file_name)
        re_img = cv2.resize(img, (TARGET_WIDTH, TARGET_HIGHTH))
        #save
        resized_name = os.path.join(target, os.path.basename(img_file_name))
        print("Target path: ", target)
        print("Saving resized image as ", resized_name)
        cv2.imwrite(resized_name, re_img)

    print("Done")
    return


def main():

    main_frame = Frame(root)
    main_frame.pack()

    path_lf = LabelFrame(main_frame, text="Path")
    path_lf.pack(side=TOP, padx=5, pady=5)
    Label(path_lf, text="Source").grid(row=0, column=0, sticky=W, padx=2, pady=2)
    source_path_entry = Entry(path_lf,textvariable=source_path, width=100)
    source_path_entry.grid(row=0, column=1, sticky=W, padx=2, pady=2)
    source_path.set(os.getcwd())

    Label(path_lf, text="Destination").grid(row=1, column=0, sticky=W, padx=2, pady=2)

    dest_path_entry = Entry(path_lf,textvariable=dest_path, width=100)
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