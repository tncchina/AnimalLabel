import os
import pandas as pd
import random
import math
import zipfile
from shutil import copyfile

#
# RAWDATALABEL_FILENAME -><filters...>-> CLEANUP_DATA_FILE -><Split and randomize>-> Training set and test set
# ex.: RawDataLabel.csv ------------> TNC512_clean_up.csv ---------> TNC512_training_random.txt, TNC512_test_random.txt
#
# ******************** Parameters session ******************************
MIN_DATA_PER_CLEANUP_CLASS = -1
MAX_DATA_PER_CLEANUP_CLASS = -1

TRAINING_TEST_RATIO = 0.80
MIN_DATA_PER_TRAINING_CLASS = 500
MAX_DATA_PER_TRAINING_CLASS = -1

ADD_HORIZONTAL_FLIPPED_IMG = True

RANDOMIZE_TRAININGDATA_ORDER = True
RANDOMIZE_TESTDATA_ORDER = True

COPY_TRAINING_PICTURE = False
COPY_TEST_PICTURE = False

OUTPUT_FILE_NAME_BASE = "TNC512"
RAWDATALABEL_FILENAME = "RawDataLabel.csv"
LOOKUP_CSV_FILENAME = 'Label_ClassID_Lookup.csv'
IMG_PACKAGE_FILENAME = "TNC512.zip"

# Define a map to rename label ( 'From', 'To')
# step 1, replacing label
Label_Rename = (('其他人员', '人'),
                ('工作人员', '人'),
                ('黄牛', '家牛')
                )
# step 2, using 'Category' value to replace following Label and Null
Label_Missing = ('未知', '待鉴定')
# step 3, removing the row with invalid label
Label_Remove = ('未知', '待鉴定', '其他')

# ******************** End of parameters ******************************


class TNCDataSet:
    raw_df = None
    lookup_df = None
    class_list = None

    def run(self):
        self.prepare_image_file()
        self.create_label_id_map()
        self.create_clean_up_data()
        self.create_train_test_data()
        self.save_training_test()
        self.create_randomized_date()
        self.create_summary()

        return

    def __init__(self):
        self.cleanup_df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self.base_folder = os.path.dirname(os.path.abspath(__file__))
        self.datasets_dir = os.path.join(self.base_folder, "DataSets")
        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)
        print("DataSets directory =>", self.datasets_dir)

        self.output_dir = os.path.join(self.base_folder, "Output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print("Output Root   =>", self.output_dir)

        self.rawdatalabel_fullname = os.path.join(self.datasets_dir, RAWDATALABEL_FILENAME)
        self.label_id_lookup_file = os.path.join(self.datasets_dir, LOOKUP_CSV_FILENAME)
        self.img_pack_fullname = os.path.join(self.datasets_dir, IMG_PACKAGE_FILENAME)
        self.img_dir = os.path.join(self.datasets_dir, "img")
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # create output file names
        self.lookup_filename = os.path.join(self.output_dir, LOOKUP_CSV_FILENAME)
        self.cleanup_fullname = os.path.join(self.output_dir, OUTPUT_FILE_NAME_BASE + "_cleanup.csv")

        self.training_split_filename = os.path.join(self.output_dir, OUTPUT_FILE_NAME_BASE + "_training_split.txt")
        self.test_split_filename = os.path.join(self.output_dir, OUTPUT_FILE_NAME_BASE + "_test_split.txt")

        self.training_data_filename = os.path.join(self.output_dir, OUTPUT_FILE_NAME_BASE + "_train.txt")
        self.test_data_filename = os.path.join(self.output_dir, OUTPUT_FILE_NAME_BASE + "_test.txt")

        self.training_data_random_filename = os.path.join(self.output_dir, OUTPUT_FILE_NAME_BASE + "_train_random.txt")
        self.test_data_random_fileName = os.path.join(self.output_dir, OUTPUT_FILE_NAME_BASE + "_test_random.txt")

        self.summary_filename = os.path.join(self.output_dir, "Summary.txt")
        return

    def _load_raw_data(self):
        if not os.path.exists(self.rawdatalabel_fullname):
            print("Raw label data file not found: ", self.rawdatalabel_fullname)
            return False
        print("Found raw label data file: ", self.rawdatalabel_fullname)
        self.raw_df = pd.read_csv(self.rawdatalabel_fullname)
        print("Raw label data loaded: ", self.raw_df.shape, self.raw_df.columns)

        print("Checking key columns")

        if 'Format' not in self.raw_df.columns:
            print("Couldn't find column 'Format")
            return False
        self.raw_df = self.raw_df[self.raw_df['Format'] == 'JPG']
        print("Total Images:", self.raw_df.shape[0])
        print("Total labels:", len(self.raw_df['Label'].unique()))
        return True

    def _extract_image_file(self):
        self.img_pack_fullname = os.path.join(self.datasets_dir, IMG_PACKAGE_FILENAME)
        if not os.path.exists(self.img_pack_fullname):
            print("Couldn't find image package file ", self.img_pack_fullname)
            print("Please download the file from \\shpeng440\TNC_RawData\TNC512.zip")
            print(" to ", self.datasets_dir, " folder, and try again.")
        if not zipfile.is_zipfile(self.img_pack_fullname):
            print("Error: ", self.img_pack_fullname, " is not a valid zip file.")
            return False
        zf = zipfile.ZipFile(self.img_pack_fullname, 'r')
        if os.path.exists(self.img_dir):
            print("Image folder exist. skip extacting images to ", self.img_dir)
        else:
            print("Unpacking image files: ", self.img_pack_fullname)
            print("Please waite...")
            zf.extractall(self.img_dir)
        print("Done!")
        return True

    def prepare_image_file(self):
        # extract all images from package
        if not self._load_raw_data():
            exit(-1)

        if not os.path.exists(self.img_pack_fullname):
            print("Error: Couldn't find image package file: ", self.img_pack_fullname)
            exit(-1)

        print("")
        print("**********************************************")
        print("********* STEP 0: Extractign image files ******")
        print("**********************************************")
        if not self._extract_image_file():
            print("Error: Failed to extract images.")
        missing_cnt = 0
        if 'FileName' not in self.raw_df.columns:
            print("Error: couldn't find 'FileName' column in label file.")
            exit(-1)
        if 'Folder' not in self.raw_df.columns:
            print("Error: couldn't find 'FileName' column in label file.")
            exit(-1)
        print("Checking images files...")
        for index, row in self.raw_df.iterrows():
            fn = row['FileName']
            folder = row['Folder']
            f = os.path.join(self.img_dir, folder, fn+".JPG")
            if not os.path.exists(f):
                print(f)
                missing_cnt += 1
        if missing_cnt > 0:
            print("Error, above ", str(missing_cnt), " file(s) are missing.")
            exit(-1)
        else:
            print("Good, we have all image files required.")
        return

    def _handle_missing_label(self):
        self.raw_df.loc[self.raw_df.Label.isnull(), ['Label']] = self.raw_df['Category']
        for r in Label_Missing:
            self.raw_df.loc[self.raw_df['Label'] == r, ['Label']] = self.raw_df['Category']
        return

    def _handle_label_replacement(self):
        for from_label, to_label in Label_Rename:
            self.raw_df.loc[self.raw_df['Label'] == from_label, 'Label'] = to_label
        return

    def _handle_label_removal(self):
        for r in Label_Remove:
            self.raw_df = self.raw_df[self.raw_df['Label'] != r]
        return

    def create_label_id_map(self):
        print("**********************************************")
        print("********* STEP 1: ID lookup file **************")
        print("**********************************************")
        self._handle_missing_label()
        self._handle_label_replacement()
        self._handle_label_removal()

        unique_labels = list(self.raw_df['Label'].unique())
        # if we Label_ClassID_Lookup exists, we use it
        if os.path.exists(self.label_id_lookup_file):
            self.lookup_df = pd.read_csv(self.label_id_lookup_file)
            print("Lookup file loaded:", self.label_id_lookup_file)
            print(self.lookup_df.shape, self.lookup_df.columns)
            print(self.lookup_df.Label)
            for label in unique_labels:
                if label.strip() not in list(self.lookup_df.Label):
                    max_id = max(self.lookup_df.ClassID)
                    print("Appending new label to lookup table")
                    print("Label: ", label, " => Assigned Class ID: ", max_id+1)
                    df = pd.DataFrame({'Label': label, 'ClassID': max_id+1})
                    self.lookup_df.append(df, ignore_index=True)
        else:
            # if we couldn't find Label_ClassID_Lookup, create a new one
            self.lookup_df = pd.DataFrame({'Label': self.raw_df['Label'].unique(),
                                           'ClassID': range(0, len(self.raw_df['Label'].unique()))})
        self.lookup_df = self.lookup_df[['Label', 'ClassID']]
        # update Label_ClassID_Lookup
        self.lookup_df.to_csv(self.label_id_lookup_file, index=False, encoding='utf-8-sig')
        # save a copy in the output folder
        self.lookup_df.to_csv(self.lookup_filename, index=False, encoding='utf-8-sig')
        print("Label to ID lookup file created: ", self.label_id_lookup_file)
        print(self.lookup_df.shape)
        # Reading the data
        self.class_list = self.lookup_df['ClassID'].unique()
        self.class_list.sort()

        min_class_id = min(self.class_list)
        max_class_id = max(self.class_list)
        print("Min Class ID: ", min(self.class_list))
        print("Max Class ID: ", max(self.class_list))
        print("Total length: ", len(self.class_list))

        if len(self.class_list) != max_class_id - min_class_id + 1:
            print("WARNING: There are ClassID not used")
        print("Done.")
        return

    def _assign_class_id(self):
        self.raw_df['ClassID'] = 0
        for row in self.lookup_df.itertuples():
            label = row[1]
            class_id = row[2]
            self.raw_df.loc[self.raw_df['Label'] == label, 'ClassID'] = class_id
        return

    def create_clean_up_data(self):
        print("**********************************************")
        print("********* STEP 2: Clean up data **************")
        print("**********************************************")
        # Adjust number of records (N) for each class
        #  min_classid<= N <= max_classid
        if 'ClassID' not in self.raw_df.columns:
            self._assign_class_id()
        for classid in self.class_list:
            label = self.lookup_df.loc[self.lookup_df['ClassID'] == classid, 'Label'].iloc[0]
            print("======Process Class ID: ", str(classid), "(", label, ") =======")
            df = self.raw_df[self.raw_df['ClassID'] == classid]
            row = df.shape[0]
            print("Rows found: ", str(row))

            if MIN_DATA_PER_CLEANUP_CLASS != -1:
                if row == 1:
                    print("Make sure each class has at least ", str(MIN_DATA_PER_CLEANUP_CLASS), " rows.")
                    temp_df = pd.DataFrame()
                    for i in range(MIN_DATA_PER_CLEANUP_CLASS):
                        temp_df = pd.concat([temp_df, df], ignore_index=True)
                    df = temp_df
                    print("Adding Row to ", str(temp_df.shape[0]))
                elif row < MIN_DATA_PER_CLEANUP_CLASS:
                    print("Make sure each class has at least ", str(MIN_DATA_PER_CLEANUP_CLASS), " rows.")
                    for i in range(MIN_DATA_PER_CLEANUP_CLASS - row):
                        srow = random.randint(1, row)
                        selected_df = df.iloc[srow - 1:srow]
                        df = pd.concat([df, selected_df], ignore_index=True)
                    print("Adding Row to ", str(df.shape[0]))

            if MAX_DATA_PER_CLEANUP_CLASS != -1:
                rowlist = []
                if row > MAX_DATA_PER_CLEANUP_CLASS:
                    print("Make sure each class has at most ", str(MAX_DATA_PER_CLEANUP_CLASS), " rows.")
                    # create a list of row number that need to be dropped.
                    while len(rowlist) < (row - MAX_DATA_PER_CLEANUP_CLASS):
                        srow = random.randint(0, row - 1)
                        if srow not in rowlist:
                            rowlist.append(srow)
                    # now delet those rows
                    df.drop(df.index[rowlist], inplace=True)
                    print("Deleting Row to ", str(df.shape[0]))

            self.cleanup_df = pd.concat([self.cleanup_df, df], ignore_index=True)
            print("Cleaned-up records: ", str(self.cleanup_df.shape[0]))

        # Saving the clean up table.
        print("Saving Cleaned-up data to ", self.cleanup_fullname)
        self.cleanup_df.to_csv(self.cleanup_fullname, index=False, encoding='utf-8-sig')
        print("Done.")
        return

    def create_train_test_data(self):
        print("**********************************************")
        print("********* STEP 3: Splitting Training/Test ******")
        print("**********************************************")

        for classid in self.class_list:
            print("======Process Class ID: ", str(classid), "=======")
            df = self.cleanup_df[self.cleanup_df['ClassID'] == classid]

            row = df.shape[0]
            split_index = math.floor(row * TRAINING_TEST_RATIO)

            if row == 1:
                self.train_df = self.train_df.append(df.iloc[0], ignore_index=True)
                self.test_df = self.test_df.append(df.iloc[0], ignore_index=True)
            elif row == 2:
                self.train_df = self.train_df.append(df.iloc[0], ignore_index=True)
                self.test_df = self.test_df.append(df.iloc[1], ignore_index=True)
            elif row >= 3:
                if split_index == row:
                    split_index = row - 1
                self.train_df = pd.concat([self.train_df, df.iloc[:split_index]], ignore_index=True)
                self.test_df = pd.concat([self.test_df, df.iloc[split_index:]], ignore_index=True)
            print("Total rows: ", row)
            print("Training  : ", split_index)
            print("Test      : ", row - split_index)

        self._copy_training_picture()
        self._copy_test_picture()

        self._generate_training_argument()

        self._adjust_training_data()

        print("Total training rows: ", self.train_df.shape[0])
        print("Total Test rows: ", self.test_df.shape[0])

        self.train_df.to_csv(self.training_split_filename, index=False, encoding='utf-8-sig')
        self.test_df.to_csv(self.test_split_filename, index=False, encoding='utf-8-sig')
        print("Done.")
        return

    def _generate_training_argument(self):
        print("**********************************************")
        print("********* STEP 3.1: Creating Training Argments ******")
        print("**********************************************")
        if ADD_HORIZONTAL_FLIPPED_IMG:
            h_flip_df = self.train_df.copy()
            h_flip_df['FileName'] = '(H)' + h_flip_df['FileName']
            self.train_df = self.train_df.append(h_flip_df, ignore_index=True)
        return

    def _copy_training_picture(self):
        if COPY_TRAINING_PICTURE:
            for label in self.train_df['Label'].unique():
                dest_dir = os.path.join(self.output_dir, 'img', 'training', label)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                    df = self.train_df[self.train_df['Label'] == label]
                    for index, row in df.iterrows():
                        source = os.path.join(self.img_dir, row['Folder'], row['FileName'] + '.JPG')
                        destination = os.path.join(dest_dir, row['FileName'] + '.JPG')
                        print(source + " => " + destination)
                        copyfile(source, destination)
        return

    def _copy_test_picture(self):
        if COPY_TEST_PICTURE:
            for label in self.test_df['Label'].unique():
                dest_dir = os.path.join(self.output_dir, 'img', 'test', label)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                    df = self.test_df[self.test_df['Label'] == label]
                    for index, row in df.iterrows():
                        source = os.path.join(self.img_dir, row['Folder'], row['FileName'] + '.JPG')
                        destination = os.path.join(dest_dir, row['FileName'] + '.JPG')
                        print(source + " => " + destination)
                        copyfile(source, destination)
        return

    def _adjust_training_data(self):
        adj_df = pd.DataFrame()
        for classid in self.class_list:
            label = self.lookup_df.loc[self.lookup_df['ClassID'] == classid, 'Label'].iloc[0]
            print("======Process Class ID: ", str(classid), "(", label, ") =======")
            df = self.train_df[self.train_df['ClassID'] == classid]
            row = df.shape[0]
            print("Rows found: ", str(row))

            if MIN_DATA_PER_TRAINING_CLASS != -1:
                if row == 1:
                    print("Make sure each class has at least ", str(MIN_DATA_PER_TRAINING_CLASS), " rows.")
                    temp_df = pd.DataFrame()
                    for i in range(MIN_DATA_PER_TRAINING_CLASS):
                        temp_df = pd.concat([temp_df, df], ignore_index=True)
                    df = temp_df
                    print("Adding Row to ", str(df.shape[0]))
                elif (row < MIN_DATA_PER_TRAINING_CLASS) and (row != 0):
                    print("Make sure each class has at least ", str(MIN_DATA_PER_TRAINING_CLASS), " rows.")
                    for i in range(MIN_DATA_PER_TRAINING_CLASS - row):
                        srow = random.randint(1, row)
                        selected_df = df.iloc[srow - 1:srow]
                        df = pd.concat([df, selected_df], ignore_index=True)
                    print("Adding Row to ", str(df.shape[0]))

            if MAX_DATA_PER_TRAINING_CLASS != -1:
                rowlist = []
                if row > MAX_DATA_PER_TRAINING_CLASS:
                    print("Make sure each class has at most ", str(MAX_DATA_PER_TRAINING_CLASS), " rows.")
                    # create a list of row number that need to be dropped.
                    while len(rowlist) < (row - MAX_DATA_PER_TRAINING_CLASS):
                        srow = random.randint(0, row - 1)
                        if srow not in rowlist:
                            rowlist.append(srow)
                    # now delete those rows
                    df.drop(df.index[rowlist], inplace=True)
                    print("Deleting Row to ", str(df.shape[0]))
            print("Adjust to:", str(df.shape[0]))
            adj_df = pd.concat([adj_df, df], ignore_index=True)
            print("Adjusted training records: ", str(adj_df.shape[0]))
        self.train_df = adj_df
        return

    def save_training_test(self):
        print("")
        print("")
        print("**********************************************")
        print("********* STEP 4: Saving Training/Test files **")
        print("**********************************************")
        df_train = pd.DataFrame(columns=['FileName', 'ID'])
        for i in range(self.train_df.shape[0]):
            folder = self.train_df.iloc[i]['Folder']
            filename = self.train_df.iloc[i]['FileName']
            f = os.path.join(self.img_dir, folder, filename+".JPG")
            df_train = df_train.append({'FileName': f, 'ID': self.train_df.iloc[i]['ClassID']}, ignore_index=True)
        df_train.to_csv(self.training_split_filename, header=False, sep='\t', index=False)
        print("Training data is saved to ", self.training_split_filename)

        df_test = pd.DataFrame(columns=['FileName', 'ID'])
        for i in range(self.test_df.shape[0]):
            folder = self.test_df.iloc[i]['Folder']
            filename = self.test_df.iloc[i]['FileName']
            f = os.path.join(self.img_dir, folder, filename+".JPG")
            df_test = df_test.append({'FileName': f, 'ID': self.test_df.iloc[i]['ClassID']}, ignore_index=True)
        df_test.to_csv(self.test_split_filename, header=False, sep='\t', index=False)
        print("Test data is saved to ", self.test_split_filename)
        print("Done.")
        return

    def create_summary(self):
        print("")
        print("")
        print("**********************************************")
        print("********* STEP 6: Summarizing  ***************")
        print("**********************************************")

        with open(self.summary_filename, "w") as summary:
            summary.write("Row Data File:\t%s\n" % self.rawdatalabel_fullname)
            summary.write("Cleaned-up file:\t%s\n" % self.cleanup_fullname)
            summary.write("Training/Total rate:\t%.3f\n" % TRAINING_TEST_RATIO)

            summary.write("Training data adjustment")
            if MAX_DATA_PER_TRAINING_CLASS != -1:
                summary.write("\tMaximum data per class: No limit\n")
            else:
                summary.write("\tMaximum data per class: " + str(MAX_DATA_PER_TRAINING_CLASS) + "\n")

            if MIN_DATA_PER_TRAINING_CLASS != -1:
                summary.write("\tMinimum data per class: No limit\n")
            else:
                summary.write("\tMinimum data per class: " + str(MIN_DATA_PER_TRAINING_CLASS) + "\n")

            if RANDOMIZE_TRAININGDATA_ORDER:
                summary.write("Training set file:\t%s\n" % self.training_data_random_filename)
            else:
                summary.write("Training set file:\t%s\n" % self.training_data_filename)

            if RANDOMIZE_TESTDATA_ORDER:
                summary.write("Testing set file:\t%s\n" % self.test_data_random_fileName)
            else:
                summary.write("Testing set file:\t%s\n" % self.test_data_filename)

            summary.write("\n\n")
            summary.write("=== Raw Data =====\n")

            summary.write("CID\tRows\n")
            for classid in list(self.class_list):
                row = self.cleanup_df[self.cleanup_df['ClassID'] == classid].shape[0]
                summary.write("%d\t%d\n" % (classid, row))
            summary.write("------------------\n")
            summary.write("Total:\t%d\n" % self.cleanup_df.shape[0])

            summary.write("\n\n")
            summary.write("==== Training Set ====\n")
            summary.write("CID\tRows\n")

            for classid in list(self.class_list):
                row = self.train_df[self.train_df['ClassID'] == classid].shape[0]
                summary.write("%d\t%d\n" % (classid, row))
            summary.write("------------------\n")
            summary.write("Total:\t%d\n" % self.train_df.shape[0])

            summary.write("\n\n")
            summary.write("==== Test Set ====\n")
            summary.write("CID\tRows\n")
            for classid in list(self.class_list):
                row = self.test_df[self.test_df['ClassID'] == classid].shape[0]
                summary.write("%d\t%d\n" % (classid, row))
            summary.write("------------------\n")
            summary.write("Total:\t%d\n" % self.test_df.shape[0])

            summary.flush()
            summary.close()
            print("Summary is created in ", self.summary_filename)
            print("Done.")
        return

    def create_randomized_date(self):
        print("")
        print("")
        print("**********************************************")
        print("********* STEP 5: Create Randomize Trainig/Test files **")
        print("**********************************************")
        if RANDOMIZE_TRAININGDATA_ORDER:
            print("Creating randomized training data.")
            f = open(self.training_split_filename, "r")
            o = open(self.training_data_random_filename, "w")
            read_buffer = []
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
                o.flush()
                o.close()
                print("Training data saved to :", self.training_data_random_filename)

        if RANDOMIZE_TESTDATA_ORDER:
            print("Creating randomized test data.")
            f = open(self.test_split_filename, "r")
            o = open(self.test_data_random_fileName, "w")
            read_buffer = []
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
                o.flush()
                o.close()
                print("Test data saved to :", self.test_data_random_fileName)
        print("Done.")
        return


def main():
    app = TNCDataSet()
    app.run()
    print("All Done!")
    return


if __name__ == '__main__':
    main()
