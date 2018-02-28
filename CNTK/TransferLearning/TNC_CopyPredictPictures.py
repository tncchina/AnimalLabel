import pandas as pd
import os
import sys
import argparse
import datetime
import zipfile
from shutil import copyfile

LOOKUP_CSV_FILENAME = 'Label_ClassID_Lookup.csv'
IMG_PACKAGE_FILENAME = "TNC512.zip"
PREDICTION_OUTPUT_FILENAME = "Test_Prediction.txt"

base_folder = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_folder, "DataSets")
img_dir = os.path.join(datasets_dir, "Img")

img_pack_fullname = os.path.join(datasets_dir, IMG_PACKAGE_FILENAME)
start_datetime = datetime.datetime.now()

prediction_df = pd.DataFrame()
output_folder = ""


def extract_img_files():
    if not os.path.exists(img_dir):
        if not os.path.exists(img_pack_fullname):
            print("Couldn't find image package file-", img_pack_fullname)
            print("Please download the file from \\shpeng440\TNC_RawData\TNC512.zip")
            sys.exit(-1)
        else:
            os.makedirs(img_dir)
            if not zipfile.is_zipfile(img_pack_fullname):
                print("Error: ", img_pack_fullname, " is not a valid zip file.")
                sys.exit(-1)
            zf = zipfile.ZipFile(img_pack_fullname, 'r')
            print("Unpacking image files: ", img_pack_fullname)
            print("Please waite...")
            zf.extractall(img_dir)
            print("Done!")
    print("Folder ", img_dir, " exists. Skip extracting image from zip file.")
    return


def read_prediction():
    global prediction_df
    prediction_df = pd.read_csv(test_prediction_output_file, header=None,
                                names=['FileName', 'ClassID', 'ClassID_Predict', 'SoftMax'], encoding='utf-8')
    return


def read_label_lookup():
    global label_id_df, output_folder
    lookup_csv_fullpath = os.path.join(output_folder, LOOKUP_CSV_FILENAME)
    label_id_df = pd.read_csv(lookup_csv_fullpath)
    return


def add_label():
    global prediction_df, label_id_df
    prediction_df = pd.merge(prediction_df, label_id_df, how='left',
                             left_on='ClassID', right_on='ClassID').drop(['ClassID'], axis=1)
    prediction_df = pd.merge(prediction_df, label_id_df, how='left',
                             left_on='ClassID_Predict', right_on='ClassID').drop(['ClassID_Predict', 'ClassID'], axis=1)
    return


def create_file_name():
    global prediction_df
    tgt_path_base = os.path.join(output_folder, 'img')
    prediction_df['tgt_path'] = tgt_path_base + "\\" + prediction_df['Label_x'] + "\\" + prediction_df['Label_y']
    prediction_df['tgt_name'] = prediction_df['FileName'].apply(lambda x: x[:-4])
    prediction_df['tgt_name'] = prediction_df['tgt_name'] + "_(" + prediction_df['SoftMax'].astype('str') + ").JPG"
    return


def copy_file():
    global prediction_df
    for folder in prediction_df['tgt_path'].unique():
        if not os.path.exists(folder):
            os.makedirs(folder)
    for r in prediction_df.itertuples():
        source_file = os.path.join(img_dir, r.FileName[:-9], r.FileName)
        print("Source => ", source_file)
        target_file = os.path.join(r.tgt_path, r.tgt_name)
        print("Target => ", target_file)
        copyfile(source_file, target_file)
    return


def main():
    global test_prediction_output_file, base_folder, output_folder
    parser = argparse.ArgumentParser(description='Copy image files to output folder '
                                                 'according to the prediciton results.'
                                                 '...\img\<True_label>\<Predicted_label>')
    parser.add_argument('output_folder')
    arg = parser.parse_args()
    # get all Output_yyyymmdd folder
    output_folder = os.path.join(base_folder, arg.output_folder)
    test_prediction_output_file = os.path.join(output_folder, PREDICTION_OUTPUT_FILENAME)
    if not os.path.exists(test_prediction_output_file):
        print("Couldn't find prediction output file- ", test_prediction_output_file)
        sys.exit(-1)
    extract_img_files()
    read_prediction()
    read_label_lookup()
    add_label()
    create_file_name()
    copy_file()
    return


if __name__ == '__main__':
    main()
