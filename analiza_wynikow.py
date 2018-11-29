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
check=[]
label = []
name = []
for folder in ["128_7ms/ok","128_7ms/nok"]:
    i = 0
    for file in os.listdir(folder):
        if(i>1000):
            continue
        if(".npy" in file):
            spectogram = np.load(folder+"/"+file)
            name.append(folder+"/"+file)
            if spectogram.shape != (128, 128): continue
            D.append((spectogram, folder))
            label.append(folder)
            check.append(spectogram) #może tu normalizować
            i+=1
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

#architektura 128x128
model = Sequential()
input_shape=(128, 128, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(1, activation='sigmoid'))

model.load_weights("weights-improvement_100model.hdf5")

prediction = []
prediction_label =[]
prediction_predicted = []
x_dataset, y_dataset = zip(*dataset)
for i in range(0,len(dataset)):
    prediction.append(model.predict(x_dataset[i].reshape(1,128,128,1))[0][0])
    prediction_label.append(y_dataset[i])
    prediction_predicted.append(round(prediction[i]))

print(prediction)
suma_srodkowychwartoci = 0
for i in range(0,len(dataset)):
    if (prediction_predicted[i]==1.0) and ("/ok" in name[i]):
        continue
    elif (prediction_predicted[i])==0.0 and ("/nok" in name[i]):
        continue
    else:
        print(name[i],prediction[i])
    if(0.40 < prediction[i] < 0.60):
        suma_srodkowychwartoci+=1
print(suma_srodkowychwartoci)