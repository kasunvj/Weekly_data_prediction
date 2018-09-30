import math
import csv
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import xlrd
import xlwt

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

workbook = xlrd.open_workbook('sampletmbz.xlsx')
worksheet = workbook.sheet_by_index(0)

raw_dataset = np.zeros((167,7),dtype= int)

for i in range (0,167):
    for j in range (0,7):
        x = worksheet.cell(i,j).value
        if (is_number(x)):
            raw_dataset[i][j]= x
        else:
            raw_dataset[i][j] = 1

print(np.shape(raw_dataset))
print(raw_dataset)




#str = "2009";
#print (str.isnumeric())
# total rows = worksheet.nows