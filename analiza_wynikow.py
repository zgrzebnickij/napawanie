import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import h5py
import itertools 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from  keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential

D=[]
D_ok = []
D_nok = []
check=[]
label = []
for folder in ["32_7ms_noise/ok","32_7ms_noise/nok"]:
    i = 0
    for file in os.listdir(folder):
        if(".npy" in file):
            spectogram = np.load(folder+"/"+file)
            if spectogram.shape != (96, 32): continue
            if(folder == "32_7ms_noise/ok"):
                D_ok.append((spectogram, folder))
                label.append(folder)
                check.append(spectogram) #może tu normalizować
                i+=1
            else:
                D_nok.append((spectogram, folder))
                label.append(folder)
                check.append(spectogram) #może tu normalizować
                i+=1
random.shuffle(D_ok)
D = D_ok
{D.append(x) for x in D_nok}
print(len(D_nok),len(D_nok[0]))
print(len(D_ok),len(D_ok[0]))
print(len(D),len(D[0]),"done")

def normalization(data,x_max,x_min):
    diff = x_max-x_min
    for rows in range(0,len(data)):
        for cols in range(0,len(data[0])):
            data[rows][cols] = (data[rows][cols]-x_min)/diff 
    return data

dataset_min = np.amin(check)
dataset_max = np.amax(check) #jak teraz znormalizuje to jak potem to zrobić?
print(dataset_min,dataset_max)
dataset_normalized = []
for data in D:
    dataset_normalized.append([normalization(data[0],dataset_max,dataset_min),data[1]])

dataset_array = keras.utils.normalize(check,axis=-1,order=2)
dataset = dataset_normalized

print(len(D))
print(len(D[0]))
print(len(D[0][0]))
print(len(D[0][0][0]))
#random.shuffle(dataset)

# train = dataset[:1000]
# test = dataset[1000:]

# x_train, y_train = zip(*train)
# x_test, y_test = zip(*test)

# # Reshape for CNN input
# x_train = np.array([x.reshape( (128, 128, 1) ) for x in x_train])
# x_test = np.array([x.reshape( (128, 128, 1) ) for x in x_test])

# # One-Hot encoding for classes
# encoder = LabelEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train)
# y_test = encoder.transform(y_test)
# print(y_test)

#architektura 96x32
model = Sequential()
input_shape=(96, 32, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(1, activation='sigmoid'))

model.load_weights("32_7ms_noise/weights-improvement_noise.hdf5")

prediction = []
prediction_label =[]
prediction_predicted = []
x_dataset, y_dataset = zip(*dataset)
for i in range(0,len(dataset)):
    prediction.append(model.predict(x_dataset[i].reshape(1,96,32,1))[0][0])
    prediction_label.append(y_dataset[i])
    prediction_predicted.append(round(prediction[i]))

#print(prediction)

for a in [[0.4,0.6],[0.35,0.65],[0.3,0.7],[0.25,0.75]]:
    suma_srodkowychwartoci = 0
    good_sum = 0
    for i in range(0,len(dataset)):
        if (prediction_predicted[i]==1.0) and ("/ok" in y_dataset[i]):
            if(a[0] < prediction[i] < a[1]):
                good_sum+=1
            continue
        elif (prediction_predicted[i])==0.0 and ("/nok" in y_dataset[i]):
            if(a[0] < prediction[i] < a[1]):
                good_sum+=1
            continue
        if(a[0] < prediction[i] < a[1]):
            #print("Wartość środkowa")
            suma_srodkowychwartoci+=1
    print(suma_srodkowychwartoci,good_sum)
# 0.36 0.65 25 wartości
#               B   G
# 0.4,0.6       18 14
# 0.35,0.65     25 21
# 0.3,0.7       34 26
# 0.25,0.75     44 38