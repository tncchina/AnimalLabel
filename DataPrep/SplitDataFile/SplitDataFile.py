# -*- coding: utf-8 -*-
"""
Syntax:
    python SplitDataFile.py
@author: Shu Peng
"""

from tkinter import *
#from tkinter import messagebox
from tkinter import filedialog

import os
#import cv2
import glob

APP_TITLE = "SplitDataFile v1.0"


class DataSplitter:
    _root = None
    _fileloading_top = None
    _main_frame = None

    _source_file = None
    _first_line = None

    _df = None

    def __init__(self):
        return

    def run(self):
        self._root = Tk()
        self._source_file = StringVar()
        self._first_line = StringVar()
        self.show_select_datafile()
        self.show_main_window()
        self._root.mainloop()
        return

    def show_select_datafile(self):
        self._root.withdraw()
        self._fileloading_top = Toplevel()
        self._fileloading_top.title(APP_TITLE)

        frame = Frame(self._fileloading_top)
        frame.pack()

        path_lf = LabelFrame(frame, text="Data File")
        path_lf.pack(side=TOP, padx=5, pady=5)

        Label(path_lf, text="Source").grid(row=0, column=0, sticky=W, padx=2, pady=2)
        source_path_entry = Entry(path_lf, textvariable=self._source_file, width=80)
        source_path_entry.grid(row=0, column=1, sticky=W, padx=2, pady=2)
        btn_browse_source = Button(path_lf, text="Browse ...", command=self.browse_path)
        btn_browse_source.grid(row=0, column=2, sticky=W, padx=2, pady=2)

        btn_load_file = Button(path_lf, text="Load", command=self.load_file)
        btn_load_file.grid(row=1, column=0, sticky=W, padx=2, pady=2)

        btn_exit = Button(path_lf, text="Exit", command=exit)
        btn_exit.grid(row=1, column=2, sticky=W, padx=2, pady=2)
        return

    def Split(self, origin_data_file):
        output_file = ""
        f = None
        o = None
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
        return

    def show_main_window(self):
        self._main_frame = Frame(self._root)
        #top.group(window=main_frame.master)
        self._main_frame.pack(padx=5, pady=5)
        Label(self._main_frame, text="First line in data file:").pack(side=TOP, padx=2, pady=2)
        firstline_entry = Entry(self._main_frame, textvariable=self._first_line)
        firstline_entry.pack(sid=TOP, padx=2, pady=2)

        opt_lf = LabelFrame(self._main_frame, text="Label Column")
        opt_lf.pack(side=TOP, padx=5, pady=5)

        return

    def browse_path(self):
        pathfilename = self._source_file.get()
        current_path = ""
        if not os.path.exists(pathfilename):
            current_path = os.getcwd()
        else:
            current_path = os.path.split(pathfilename)[0]
        answer = filedialog.askopenfilename(title="Select a data file", initialdir=current_path,
                                            filetypes=[('Text file', '*.txt'), ('CSV file', '*.csv')])
        if answer:
            self._source_file.set(answer)
        return

    def load_file(self):
        fullpathfilename = self._source_file.get()
        if not os.path.exists(fullpathfilename):
            print("Error: file not found-", fullpathfilename)
            return
        datapath, datafile = os.path.split(fullpathfilename)
        print("Path:", datapath)
        print("File:", datafile)
        self._fileloading_top.destroy()
        title = "Data file: " + self._source_file.get()
        self._root.title(title)
        self._root.deiconify()
        return


def main():
    app = DataSplitter()
    app.run()
    return

if __name__ == '__main__':
    main()
