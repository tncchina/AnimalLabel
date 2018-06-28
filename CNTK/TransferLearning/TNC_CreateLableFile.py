import os
import pandas as pd

# Switches to control whether we want to list following flipped image on list
ADD_HORIZONTALLY_FLIPPED_IMG = False
ADD_VERTICALLY_FLIPPED_IMG = False
ADD_BOTH_FLIPPED_IMG = False

base_folder = os.path.dirname(os.path.abspath(__file__))
data_sets_dir = os.path.join(base_folder, "DataSets")
source_path = os.path.join(data_sets_dir, "ByLabelsExt")
data_label_file = os.path.join(data_sets_dir, "RawDataLabelsExt.csv")

df = pd.DataFrame(columns=['FileName', 'Format', 'Folder', 'Category', 'Label'])
index = 0
for root, dirs, files in os.walk(source_path):
    for name in files:
        folder_name = name[(name.find(')') + 1):name.rfind('-')]
        head, tail = os.path.split(root)

        prefix = name[:3]
        if prefix == '(H)':
            # Horizontally flipped image
            if ADD_HORIZONTALLY_FLIPPED_IMG:
                df.loc[index] = [name[:-4], 'JPG', folder_name, tail, tail]
                index += 1
        elif prefix == '(V)':
            # Vertically flipped image
            if ADD_VERTICALLY_FLIPPED_IMG:
                df.loc[index] = [name[:-4], 'JPG', folder_name, tail, tail]
                index += 1
        elif prefix == '(B)':
            # Flipped both vertically and horizontally
            if ADD_BOTH_FLIPPED_IMG:
                df.loc[index] = [name[:-4], 'JPG', folder_name, tail, tail]
                index += 1
        else:
            # This is original image
            df.loc[index] = [name[:-4], 'JPG', folder_name, tail, tail]
            index += 1
        print(os.path.join(root, name))
        print("Name        = ", os.path.basename(name))
        print("File Name   = ", name[:-4])
        print("Folder Name = ", folder_name)
        print("Label       = ", tail)
        print("==================================================")
        print("")

print(df.shape)
df.to_csv(data_label_file, index=False, encoding='utf-8-sig')
g = pd.DataFrame({'Count': df.groupby([ "Label"] ).size()}).reset_index()
g = g.sort_values(by=["Count"], ascending=False)
print(g)

