# -*- coding: utf-8 -*-
"""
Syntax:
    python SplitDataFile.py
@author: Shu Peng
"""

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

import os
import cv2
import glob

import glob

root = Tk()
APP_TITLE = "SplitDataFile v1.0"

def load_file(data_file):
    return

def Split(origin_data_file):
    output_file = ""
    try:
        f = open(origin_data_file, "r")
        o = open(output_file, "w")
    except FileNotFoundError:
        print("File not found: ", origin_data_file)
        exit(-1)

    read_buffer = []
    total_lines = 0
    if f and o:
        for line in f:
            read_buffer.append(line)
        f.close()

        total_lines = len(read_buffer)

        while total_lines > 0:
            random_line = random.randint(0, total_lines - 1)
            o.write(read_buffer[random_line])
            del read_buffer[random_line]
            total_lines -= 1


source_file = StringVar()

def browse_path():
    pathfilename = source_file.get()
    current_path = ""
    if not os.path.exists(pathfilename):
        current_path = os.getcwd()
    else:
        current_path = os.path.split(pathfilename)[0]
    answer = filedialog.askopenfilename(title="Select a data file", initialdir=current_path,
                                        filetypes = [('Text file','*.txt'),('CSV file', '*.csv')])
    if answer:
        source_file.set(answer)
    return


def main():
    main_frame = Frame(root)
    main_frame.master.title(APP_TITLE)
    main_frame.pack()

    path_lf = LabelFrame(main_frame, text="Data File")
    path_lf.pack(side=TOP, padx=5, pady=5)
    Label(path_lf, text="Source").grid(row=0, column=0, sticky=W, padx=2, pady=2)
    source_path_entry = Entry(path_lf,textvariable=source_file, width=80)
    source_path_entry.grid(row=0, column=1, sticky=W, padx=2, pady=2)

    btn_browse_source = Button(path_lf, text="Browse ...", command=browse_path)
    btn_browse_source.grid(row=0, column=2, sticky=W, padx=2, pady=2)

    root.mainloop()
    return

if __name__ == '__main__':
    main()