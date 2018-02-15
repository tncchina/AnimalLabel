"""
Class name: ConfusionMatrix.py

2018.1.1
Shu Peng

Usage: 1. Used as object
           cm = ConfusionMatrix()

           # (optional)
           cm.set_id_lookup_file('Label-ID_lookup.txt')
           cm.add_result( true_value, prediction)
           ...
           cm.change_id_to_label()
           cm.save_matrix()

           cm.load_matrix()

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


class ConfusionMatrix:
    _min_class_id = 0
    _max_class_id = 0

    _correct_cnt = 0

    _m = None
    _m_df = None

    _lookup_file = ""
    _lookup_df = None

    _true = []
    _pred = []

    def __init__(self, number_of_classes = 2):
        if number_of_classes < 2:
            print("Error, number of classes should be at least 1.")
            self._max_class_id = 1
        else:
            self._max_class_id = number_of_classes -1
        self._m = np.zeros([self._max_class_id+1, self._max_class_id+1])
        return

    def set_id_lookup_file(self, filename):
        self._lookup_file = filename
        if not os.path.exists(self._lookup_file):
            print("Lookup file not found -", self._lookup_file)
            return
        else:
            self._lookup_df = pd.read_csv(self._lookup_file)
        return

    def add_result(self, true_value, prediction):
        if type(true_value)!= int or type(prediction) != int:
            print("Error, both true class ID and predicted class ID should be integer.")
            return
        self._true.append(true_value)
        self._pred.append(prediction)

        if true_value > prediction:
            smaller = prediction
            bigger = true_value
        else :
            smaller = true_value
            bigger = prediction

        if smaller < self._min_class_id:
            new_upper_left = self._min_class_id - smaller
            v = np.zeros([new_upper_left, self._max_class_id - self._min_class_id + 1])
            self._m = np.vstack((v, self._m))
            h = np.zeros([self._max_class_id - smaller + 1, new_upper_left])
            self._m = np.hstack((h,self._m))
            self._min_class_id = smaller

        if bigger > self._max_class_id:
            new_lower_right = bigger - self._max_class_id
            v = np.zeros([new_lower_right, self._max_class_id - self._min_class_id + 1])
            self._m = np.vstack((self._m, v))
            h = np.zeros([bigger - self._min_class_id + 1, new_lower_right ])
            self._m = np.hstack((self._m, h))
            self._max_class_id = bigger

        self._m[true_value-self._min_class_id][prediction-self._min_class_id] += 1
        if true_value == prediction:
            self._correct_cnt += 1
        return

    def print_matrix(self):
        print("Confusion Matrix:")
        if self._m_df is None:
            print("Class ID: ", self.class_id_list())
            print("Accuracy: %f" % self.accuracy())
            print(self._m)
            kappa = "Cohen's Kappa score is " + str(cohen_kappa_score(self._true, self._pred))
            print(kappa)
        else:
            self._m_df
        return

    def get_min_class_id(self):
        return self._min_class_id

    def get_max_class_id(self):
        return self._max_class_id

    def class_id_list(self):
        return [i for i in range(self._min_class_id, self._max_class_id+1)]

    def accuracy(self):
        if self._m.sum() == 0:
            print("Error, no result added yet.")
            return 0.0
        else:
            return self._correct_cnt/self._m.sum()

    def savetxt(self, text_file_name):
        headerlines = '=== Confusion Matrix ===\r\nClass ID: {0:}\r\nAccuracy: {1:f}\r\n'.format\
            (str(self.class_id_list()), self.accuracy())
        np.savetxt(text_file_name, self._m, '%.0d',
                   header=headerlines,
                   footer='========================',
                   newline='\r\n')
        kappa = "Cohen's Kappa score is " + str(cohen_kappa_score(self._true, self._pred))+ "\r\n"
        with open(text_file_name,'a') as f:
            f.write(kappa)
        if self._m_df is not None:
            self._m_df.to_excel(text_file_name+".xlsx")
        return

    def change_id_to_label(self):
        if self._lookup_df is not None:
            number_of_ids = self._max_class_id - self._min_class_id + 1
            self._m_df = pd.DataFrame(self._m, columns=self._lookup_df['Label'][:number_of_ids])
            self._m_df['Label']=self._lookup_df['Label'][:number_of_ids]
            re_order_column = ['Label']+ list(self._m_df.columns[:-1])
            self._m_df=self._m_df[re_order_column]
        return


def main():
    cm = ConfusionMatrix()
    cm.print_matrix()
    cm.add_result(-1,-1)
    cm.print_matrix()
    cm.add_result(1,2)
    cm.print_matrix()
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.add_result(5,2)
    cm.set_id_lookup_file("G:\GitProjects\AnimalLabel\CNTK\TransferLearning\Output\Label_ClassID_Lookup.csv")
    cm.change_id_to_label()
    cm.print_matrix()
    cm.savetxt("test.txt")


if __name__ == '__main__':
    main()






