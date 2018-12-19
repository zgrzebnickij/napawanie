from  keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
from scipy import signal
import h5py
import numpy as np
import matplotlib.pyplot as plt
import threading

class classification():
    def __init__(self):
        low = 0.23
        high = 0.8
        #self.i=0
        self.b ,self.a = signal.butter(4,(low,high) , btype="bandpass")  ## filtr pasmowo przepustowy 
        self.num_of_section = 32
        self.time_duration = 7 ##ms
        self.sr=40000 #Hz
        self.samples_for_one_col = int(0.001*self.time_duration*self.sr)
        self.samples_for_spektrogram = int(self.samples_for_one_col*self.num_of_section)
        self.ln=self.samples_for_spektrogram/self.num_of_section
        self.make_model((96,32,1))
        self.model.load_weights("./GUI/models/final_model_fold_val_acc_12_2_weights.hdf5")
    
    def set_max_min(self,maxi,mini):
        self.maxi=maxi
        self.mini=mini

    def make_model(self,input_shape):    
        self.model = Sequential()

        self.model.add(Conv2D(12, (5, 5), strides=(1, 1), input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(12, (5, 5), padding="valid"))
        self.model.add(MaxPooling2D((4, 2), strides=(4, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(12, (5, 5), padding="valid"))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(1, activation='sigmoid'))

    def normalization(self,data):
        diff = self.maxi-self.mini
        for rows in range(0,len(data)):
            for cols in range(0,len(data[0])):
                data[rows][cols] = (data[rows][cols]-self.mini)/diff 
        return data

    def prepare_spectrogram(self,sound):
        output_signal = signal.filtfilt(self.b, self.a, sound)
        f, t, ps = signal.stft(output_signal, self.sr, nperseg=self.ln,noverlap=0,boundary=None)
        start_from = 25
        normalized = self.normalization(np.abs(ps[start_from:start_from + 96]))
        #if(ps.shape==(141,self.num_of_section)):
            #threading.Thread(target=self.savePlot(t,f[start_from:start_from + 96], np.abs(ps[start_from:start_from + 96]),self.i)).start()
            #self.i+=1
            #np.save(path_to_file,np.abs(ps[start_from:start_from + 128]))
        return normalized

    def get_preditction(self,input_array):
        spektrogram = self.prepare_spectrogram(input_array)
        prediction = self.model.predict(spektrogram.reshape(1,96,32,1))
        return prediction[0]

    def savePlot(self,t,f,ps,i):
        plt.pcolormesh(t,f,ps ,cmap='gray_r')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        cbar = plt.colorbar()
        cbar.set_label("Intensity (dB)")
        plt.savefig("nagrania/nagrania{0}.png".format(i))
        plt.close()

        

