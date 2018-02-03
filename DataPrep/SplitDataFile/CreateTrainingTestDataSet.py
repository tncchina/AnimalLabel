import os
import pandas as pd
import random
import math

# ******************** Parameters session ******************************
Training_Test_Data_Ratio = 0.80
Minimum_Data_Per_Class = 20
Maximum_Data_Per_Class = -1

Random_Training_Data_Order = True
Random_Test_Data_Order = True

Raw_Data_File = "RawDataLabel.csv"

# ******************** End of parameters ******************************

class RawData:
    def __init__(self):
        self.base_folder = os.path.dirname(os.path.abspath(__file__))
        self.datasets_dir = os.path.join(self.base_folder, "..", "..", "CNTK", "TransferLearning", "DataSets")
        self.DataFileRoot = os.path.join(self.datasets_dir, "TNC2")
        print("DataSets Root:", self.datasets_dir)

        # create output file names
        self.Raw_Data_PathFileName = os.path.join(self.DataFileRoot, Raw_Data_File)
        filename, ext = os.path.splitext(self.Raw_Data_File)
        CleanUp_Data_File = filename + "_cleanup" + ext
        self.CleanUp_Data_PathFileName = os.path.join(self.DataFileRoot, CleanUp_Data_File)

        Training_Data_File = filename + "_train.txt"
        self.Training_Data_PathFileName = os.path.join(self.DataFileRoot, Training_Data_File)

        Training_Data_Random_File = filename + "_train_random.txt"
        self.Training_Data_Random_PathFileName = os.path.join(self.DataFileRoot, Training_Data_Random_File)

Test_Data_File = filename + "_test.txt"
Test_Data_PathFileName = os.path.join(DataFileRoot, Test_Data_File)

Test_Data_Random_File = filename + "_test_random.txt"
Test_Data_Random_PathFileName = os.path.join(DataFileRoot, Test_Data_Random_File)

Summary_PathFileName = os.path.join(DataFileRoot, "Summary.txt")

print("")
print("")
print("**********************************************")
print("********* STEP 0: Make sure data exists ******")
print("**********************************************")

name_id_df = pd.read_csv(Raw_Data_PathFileName)
print(name_id_df.dtypes)

print("Checking images files...")
missing_cnt = 0
for f in name_id_df['FilePathName']:
    if not os.path.exists(f):
        print(f)
        missing_cnt+=1
if missing_cnt > 0:
    print("Error, above ", str(missing_cnt), " file(s) are missing.")
    exit(-1)

# Reading the data
classlist = name_id_df['Class_ID'].unique()
classlist.sort()

min_classid = min(classlist)
max_classid = max(classlist)
print("Min Class ID: ", min(classlist))
print("Max Class ID: ", max(classlist))
print("Total length: ", len(classlist))

if len(classlist) != max_classid-min_classid+1:
    print("Error: There are missing Class ID(s)")
    exit(-1)

#
temp_df = pd.DataFrame()
cleanup_df = pd.DataFrame()

print("**********************************************")
print("********* STEP 1: Clean up data **************")
print("**********************************************")
# Adjust number of records (N) for each class
#  min_classid<= N <= max_classid
for classid in classlist:
    print("======Process Class ID: ", str(classid), "=======")
    df = name_id_df[name_id_df['Class_ID'] == classid]
    row = df.shape[0]
    print("Rows found: ", str(row))

    if Minimum_Data_Per_Class != -1:
        if row == 1:
            print("Make sure each class has at least ", str(Minimum_Data_Per_Class), " rows.")
            for i in range(Minimum_Data_Per_Class):
                temp_df = pd.concat([temp_df, df], ignore_index=True)
            df = temp_df
            print("Adding Row to ", str(df.shape[0]))
        elif row < Minimum_Data_Per_Class:
            print("Make sure each class has at least ", str(Minimum_Data_Per_Class), " rows.")
            for i in range(Minimum_Data_Per_Class - row):
                srow = random.randint(1, row)
                selected_df = df.iloc[srow - 1:srow]
                df = pd.concat([df, selected_df], ignore_index=True)
            print("Adding Row to ", str(df.shape[0]))

    if Maximum_Data_Per_Class != -1:
        rowlist = []
        if row>Maximum_Data_Per_Class:
            print("Make sure each class has at most ", str(Maximum_Data_Per_Class), " rows.")
            # create a list of row number that need to be dropped.
            while len(rowlist) < (row-Maximum_Data_Per_Class):
                srow = random.randint(0, row-1)
                if srow not in rowlist:
                    rowlist.append(srow)
            #now delet those rows
            df.drop(df.index[rowlist], inplace=True)
            print("Deleting Row to ", str(df.shape[0]))

    cleanup_df = pd.concat([cleanup_df, df], ignore_index=True)
    print("Cleaned-up records: ", str(cleanup_df.shape[0]))

#Saving the clean up table.
print("Saving Cleaned-up data to ", CleanUp_Data_PathFileName)
cleanup_df.to_csv(CleanUp_Data_PathFileName, index=False)

print("")
print("")
print("**********************************************")
print("********* STEP 2: Spliting Trainig/Test ******")
print("**********************************************")

train_df = pd.DataFrame()
test_df = pd.DataFrame()
for classid in classlist:
    print("======Process Class ID: ", str(classid), "=======")
    df = cleanup_df[cleanup_df['Class_ID'] == classid]
    row = df.shape[0]
    split_index = math.floor(row * Training_Test_Data_Ratio)
    train_df = pd.concat([train_df, df.iloc[:split_index]], ignore_index=True)
    test_df = pd.concat([test_df, df.iloc[split_index:]], ignore_index=True)
    print("Training:", split_index)
    print("Test:", row-split_index)

    print("Total training rows: ", train_df.shape[0])
    print("Total Test rows: ", test_df.shape[0])

print("")
print("")
print("**********************************************")
print("********* STEP 3: Saving Trainig/Test files **")
print("**********************************************")

train_df.to_csv(Training_Data_PathFileName, header=False, sep='\t', index=False)
test_df.to_csv(Test_Data_PathFileName, header=False, sep='\t', index=False)

if Random_Training_Data_Order:
    print("Creating randomized training data.")
    f = open(Training_Data_PathFileName, "r")
    o = open(Training_Data_Random_PathFileName, "w")
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
        o.flush()
        o.close()
        print("Training data saved to :", Training_Data_Random_PathFileName)

if Random_Test_Data_Order:
    print("Creating randomized test data.")
    f = open(Test_Data_PathFileName, "r")
    o = open(Test_Data_Random_PathFileName, "w")
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
        o.flush()
        o.close()
        print("Test data saved to :", Test_Data_Random_PathFileName)

print("")
print("")
print("**********************************************")
print("********* STEP 4: Summerizing  ***************")
print("**********************************************")

with open(Summary_PathFileName, "w") as summary:
    summary.write("Row Data File:\t%s\n" % Raw_Data_File)
    summary.write("Cleaned-up file:\t%s\n" % CleanUp_Data_File)
    summary.write("Training/Total rate:\t%.3f\n" % Training_Test_Data_Ratio)
    if Random_Training_Data_Order:
        summary.write("Training set file:\t%s\n" % Training_Data_Random_File)
    else:
        summary.write("Training set file:\t%s\n" % Training_Data_File)
    if Random_Test_Data_Order:
        summary.write("Testing set file:\t%s\n" % Test_Data_Random_File)
    else:
        summary.write("Testing set file:\t%s\n" % Test_Data_File)

    summary.write("\n\n")
    summary.write("=== Raw Data =====\n")

    summary.write("CID\tRows\n")
    for classid in classlist:
        row = cleanup_df[cleanup_df['Class_ID'] == classid].shape[0]
        summary.write("%d\t%d\n" % (classid,row))
    summary.write("------------------\n")
    summary.write("Total:\t%d\n" % cleanup_df.shape[0])

    summary.write("\n\n")
    summary.write("==== Training Set ====\n")
    summary.write("CID\tRows\n")
    for classid in classlist:
        row = train_df[train_df['Class_ID'] == classid].shape[0]
        summary.write("%d\t%d\n" % (classid, row))
    summary.write("------------------\n")
    summary.write("Total:\t%d\n" % train_df.shape[0])

    summary.write("\n\n")
    summary.write("==== Test Set ====\n")
    summary.write("CID\tRows\n")
    for classid in classlist:
        row = test_df[test_df['Class_ID'] == classid].shape[0]
        summary.write("%d\t%d\n" % (classid,row))
    summary.write("------------------\n")
    summary.write("Total:\t%d\n" % test_df.shape[0])

    summary.flush()
    summary.close()
print("Done!")