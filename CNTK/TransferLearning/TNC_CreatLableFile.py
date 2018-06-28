import os
import pandas as pd

base_folder = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_folder, "DataSets")
source_path = os.path.join(datasets_dir, "ByLabelsExt")
data_label_file = os.path.join(datasets_dir, "RawDataLabelsExt.csv")

df = pd.DataFrame(columns=['FileName', 'Format', 'Folder', 'Category', 'Label'])
index = 0
for root, dirs, files in os.walk(source_path):
    for name in files:
        print(os.path.join(root,name))
        print("Name        = ", os.path.basename(name))
        print("File Name   = ", name[:-4])
        folder_name = name[(name.find(')')+1):name.rfind('-')]
        print("Folder Name = ", folder_name)
        head, tail = os.path.split(root)
        print("Label       = ", tail)
        print("==================================================")
        print("")
        df.loc[index] = [name[:-4], 'JPG', folder_name, tail, tail]
        index += 1

print(df.shape)
df.to_csv(data_label_file, index=False, encoding='utf-8-sig')
g = pd.DataFrame({'Count': df.groupby([ "Label"] ).size()}).reset_index()
g = g.sort_values(by=["Count"], ascending=False)
print(g)

