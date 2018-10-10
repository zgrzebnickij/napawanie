import librosa
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal

class data_prepare:
    number_of_cracks = 0

    def open_csv_and_save_to_wav(self,path,file):
        with open(path+file, 'r',newline='\n',encoding="Latin-1") as csvfile:
            sound = csv.reader(csvfile, delimiter='\t')
            print(sound.line_num)
            channel_num = 0
            try:
                array_two_chanels = []
                for channel in sound:
                    array = []
                    for probe in channel:
                        array.append(float(probe.replace(',','.')))
                    print(len(array))
                    array_two_chanels.append(np.array(array))
                    channel_num+=1
                array_two_chanels = np.array(array_two_chanels).reshape(-1,2)
                print("loaded sound with {0} chanels".format(channel_num))
                return array_two_chanels
            except csv.Error as e:
                print("Can't load",'path %s, line %d: %s' % (path, sound.line_num, e))
                return []

def widmo(y):
    f, Pwelch_spec = signal.welch(y, 40000, scaling='spectrum')
    plt.semilogy(f, Pwelch_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()

def Wykres_czasowy(y):
    x = [x/40000 for x in range(len(y))]
    plt.plot(x,y)
    plt.show()    

data_opener = data_prepare()
for folder in ['Pękanie 4']: #,'Pękanie 2','Pękanie 3','Pękanie 4'
    for folder2 in os.listdir(folder):
        cracks = np.array([])
        sound = np.array([])
        for file in os.listdir(folder+'/'+folder2):
            path_to_file = "{0}/{1}/".format(folder,folder2)
            print(path_to_file+file)
            if("6_dźwięk.csv" in file):
                sound = np.array(data_opener.open_csv_and_save_to_wav(path=path_to_file,file=file))
                if(sound==[]):
                    break
                sound = sound.reshape(2,-1)
                y = sound[0]
                amp = max(y)
                print(amp)
                widmo(y)
                b, a = signal.butter(4, 0.35, 'high') # 0.3
                b ,a = signal.butter(4,(0.3,0.8) , btype="bandpass")
                output_signal = signal.filtfilt(b, a, y)
                widmo(output_signal)
                Wykres_czasowy(output_signal)
                f, t, Zxx = signal.stft(output_signal, 40000, nperseg=400) # signal.spectrogram(y, 40000,nperseg=1000)
                print(f.shape,t.shape,Zxx.shape)
                amp = max(output_signal)
                print(amp)
                plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp/2) 
                plt.title('STFT Magnitude')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.show()
                librosa.output.write_wav(path_to_file+file.replace("_dźwięk.csv","_nagranie_filtrowane.wav"),np.array(output_signal,dtype='float'),40000,norm=True)
                print(sound)