"""
 Image Organizer 1.0
 Syntax
 ImgOrg -r Training_result_dir -s image_source_dir -d image_destination_dir
"""

import sys
import os
import getopt
import pandas as pd
from  shutil import copyfile
LOOKUP_FILENAME = 'Label_ClassID_Lookup.csv'
TRAINING_FILENAME =
class ImgOrg:
    def __init__(self):
        self.base_dir =  os.path.dirname(os.path.realpath(__file__))
        self.output_dir = os.path.join(self.base_dir, "Output")

    def set_result_dir(self, folder):
        self.output_dir = folder
        return

    def set_img_dir(self, folder):
        self.img_dir = folder
        return

    def set_img_pkg_file(self, file):
        self.img_pkg_file = file
        return

    def set_dest_dir(self, folder):
        self.set_dest_dir = folder
        return

    def _load_lookup_table(self):
        full_path = os.path.join(self.output_dir, LOOKUP_FILENAME)

    def _load_training_set(self):

    def _load_test_set(self):


    def run(self):
        self._load_lookup_table()
        self._load_training_set()
        self._load_test_set()
        return



def print_help():
    print("")

def main(argv):
    print(__file__)
    imgorg = ImgOrg()
    try:
        opts, args = getopt.getopt(argv,"r:s:d:",["rDir=","sDir=","dDir="])
    except getopt.GetoptError:
      print_help()
      sys.exit(2)

    opt_cnt = 0
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-r", "--rDir"):
            imgorg.set_result_dir(arg)
            opt_cnt += 1
        elif opt in ("-s", "--sDir"):
            imgorg.set_img_dir(arg)
            opt_cnt += 2
        elif opt in ("-d", "--dDir"):
            imgorg.set_dest_dir(arg)
            opt_cnt += 4
    if opt_cnt == 7:
        imgorg.run()
    else:
        print("Error - missing paramenter(s).")
        sys.exit()

if __name__ == "__main__":
   main(sys.argv[1:])