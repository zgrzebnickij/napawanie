import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import h5py
import itertools 
from  keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
import time
from scipy import signal

#Policzenie czsu dla różnych wielkości okna

low = 0.23
high = 0.8
b ,a = signal.butter(4,(low,high) , btype="bandpass")  ## filtr pasmowo przepustowy 


def normalization(data,x_max,x_min):
    diff = x_max-x_min
    for rows in range(0,len(data)):
        for cols in range(0,len(data[0])):
            data[rows][cols] = (data[rows][cols]-x_min)/diff 
    return data

def Prepare_spektrogram(data,y_size,num_of_section,number_of_samples):
    ln=number_of_samples/num_of_section
    output_signal = signal.filtfilt(b, a, data)
    f, t, ps = signal.stft(output_signal, sr, nperseg=ln,noverlap=0,boundary=None)
    #print(ps.shape,ln,num_of_section,y_size) 
    normalized = normalization(np.abs(ps)  ,0.0006509951221093928,7.295001191662265e-14)[0:y_size]
    return normalized

def making_model(input_shape):

    model = Sequential()

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(1, activation='sigmoid'))
    return model


cala = []
for j in range(124,32,-8):#range(100,132,4):#range(128,96,-4):
    wiersz = []
    for num_of_section in  range(124,32,-8):#range(36,132,8):#range(128,32,-4):
        input_size=(j, num_of_section, 1)
        try:
            model = making_model(input_size)
        except:
            print(j,num_of_section)
            print("imposible architecture")
            break
        time_duration = 7 ##ms
        sr=40000 #Hz
        samples_for_one_col = int(0.001*time_duration*sr)
        samples_for_spektrogram = int(samples_for_one_col*num_of_section)
        x = np.random.randn(samples_for_spektrogram)
        #print("ilość sekcji {0}".format(num_of_section),"próbki dla jednej kolumny {0}".format(samples_for_one_col),"ilość próbek {0}".format(samples_for_spektrogram))
        suma=0
        for m in range(10):
            start = time.time()
            spektogram = Prepare_spektrogram(x,j,num_of_section,samples_for_spektrogram)
            prediction = model.predict(spektogram.reshape(1,j,num_of_section,1))
            end = time.time()
            if(m!=0):
                suma+=end-start
            #print(end-start)
        wiersz.append(suma/9)
        #print(end - start)
    cala.append(wiersz)
    print(wiersz,j)
rows = [n for n in range(124,32,-8)]
cols = [m for m in range(124,32,-8)]
print(cols,rows)
plt.pcolormesh(cols,rows,cala,shading='gouraud')
plt.xlabel("Numbers of columns")
plt.ylabel("Number of fft samples")
plt.title("Time for calculating one spectrogram")
cbar = plt.colorbar()
cbar.set_label("seconds")
plt.show()

# input_size=(100,100, 1)
# model = making_model(input_size)
# time_duration = 7 ##ms
# sr=40000 #Hz
# samples_for_one_col = int(0.001*time_duration*sr)
# samples_for_spektrogram = int(samples_for_one_col*100)
# x = np.random.randn(samples_for_spektrogram)
# #print("ilość sekcji {0}".format(num_of_section),"próbki dla jednej kolumny {0}".format(samples_for_one_col),"ilość próbek {0}".format(samples_for_spektrogram))

# while(1):
#     start = time.time()
#     spektogram = Prepare_spektrogram(x,100,100,samples_for_spektrogram)
#     prediction = model.predict(spektogram.reshape(1,100,100,1))
#     end = time.time()
#     print(end-start)