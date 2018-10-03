import math
import csv
import numpy as np
#from sklearn import preprocessing
#import tensorflow as tf
#from tensorflow import keras
#rom tensorflow.keras.callbacks import TensorBoard
from keras.models import  Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import  array


import xlrd
import xlwt
from matplotlib import pyplot as plt


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def create_dataset(raw_dataset, look_back):
    dataX,dataY = [],[]
    for i in range (len(raw_dataset)-look_back-1):
        dataX.append(raw_dataset[i:i+look_back,0])  # ( data x is (maximum range)* (look_back)  <<<Input has look_back elements
        dataY.append(raw_dataset[i+look_back,0]) # data y is (maximum range) * (1)    <<< Output has one element
    print(np.shape(dataX))
    return np.array(dataX),np.array(dataY)

workbook = xlrd.open_workbook('sampletmbz.xlsx')
worksheet = workbook.sheet_by_index(0)
row_count = worksheet.nrows
raw_dataset = np.zeros((row_count,1),dtype= int)

for i in range (0,row_count):
    x = worksheet.cell(i,0).value
    if (is_number(x)):
        raw_dataset[i][0]= x
    else:
        raw_dataset[i][0] = 50


#print(np.shape(raw_dataset))
#print(raw_dataset)
#plt.plot(raw_dataset[:,2])
#plt.show()

np.random.seed(7)
raw_dataset = raw_dataset.astype('float32')
#print(raw_dataset)

scaler = MinMaxScaler(feature_range=(0,1))
raw_dataset = scaler.fit_transform(raw_dataset)
#print(raw_dataset)

train_size = int(len(raw_dataset)*.7)
test_size = len(raw_dataset)- train_size
train ,test = raw_dataset[0:train_size,:],raw_dataset[train_size:len(raw_dataset),:]
#print(len(train),len(test))

look_back = 8
trainX, trainY = create_dataset(train,look_back)
testX, testY = create_dataset(test,look_back)

# Actual Prediction
in1 =np.zeros((look_back,1),float)
in1 =[]

in1.append(raw_dataset[len(raw_dataset)-look_back:len(raw_dataset),0]) #<< in1 has look_back_elements

in2 =np.array(in1)


print(np.shape(testX))
print(np.shape(in1))
print("-")

trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1, testX.shape[1]))
in2 = np.reshape(in2,(in2.shape[0],1,in2.shape[1]))

print(np.shape(testX))
print(np.shape(in2))

model = Sequential()
model.add(LSTM(6,input_shape=(1,look_back)))

model.add(Dense(1,use_bias= True))
model.add(Dense(6,use_bias= True))
model.add(Dense(6,use_bias= True))
model.add(Dense(1,use_bias= True))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=100,batch_size=1,verbose=2)



trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
actualPredict= model.predict(in2)


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
actualPredict = scaler.inverse_transform(actualPredict)

trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train Score: %0.2f RMSE'%(trainScore))
testScore =math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score :%0.2f RMSE'%(testScore))

print("Predict: ",int(actualPredict))
#in1 =np.zeros((1,1),float)
#in1[0][0] =20
#in1 =scaler.fit_transform(in1)
#in1  = np.reshape(in1,(in1.shape[0],1,in1.shape[1]))
#print(scaler.inverse_transform(model.predict(in1)))


# shift train predictions for plotting
trainPredictPlot = np.empty_like(raw_dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(raw_dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(raw_dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(raw_dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()







#workbook_write = xlwt.Workbook(encoding = 'ascii')
#worksheet_write = workbook_write.sheet_by_index(0)
#worksheet.write(row_count+1,0,day_prediction )


#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#str = "2009";
#print (str.isnumeric())
# total rows = worksheet.nows