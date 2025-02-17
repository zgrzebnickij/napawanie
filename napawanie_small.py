import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import csv 
import numpy as np
import os
import random
#import keras
from scipy import signal
from scipy.io.wavfile import write
from acoustics import generator


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
                    librosa.output.write_wav(path+file.replace("_dźwięk.csv","_nagranie.wav"),np.array(array,dtype='float'),40000,norm=True)
                    channel_num+=1
                array_two_chanels = np.array(array_two_chanels).reshape(-1,2)
                print("loaded sound with {0} chanels".format(channel_num))
                return array_two_chanels
            except csv.Error as e:
                print("Can't load",'path %s, line %d: %s' % (path, sound.line_num, e))
                return []

    def open_csv_and_count(self,path,file):
        with open(path+file, 'r',newline='\n',encoding="Latin-1") as csvfile:
            sound = csv.reader(csvfile, delimiter='\t')
            num_of_cracks = 0
            try:
                array = []
                for channel in sound:
                    for probe in channel[0].split("\t"):
                        array.append(float(probe.replace(',','.')))
                        num_of_cracks+=1
                    #print(array)
                self.number_of_cracks+=num_of_cracks
                print("loaded sound with {0} chanels".format(num_of_cracks))
                return array
            except csv.Error as e:
                print("Can't load",'path %s, line %d: %s' % (path, sound.line_num, e))

def saving_spectograms(sound,sr,ln,file,label):
    #plt.figure()
    ps = librosa.feature.melspectrogram(y=sound, sr=sr,hop_length=ln+1)
    print("max:",np.amax(ps),"min:",np.amin(ps))
    if(ps.shape==(128,128)):
        np.save(label+"/spectogram_"+file.replace("/","_"),ps)
        # librosa.display.specshow(keras.utils.normalize(ps,axis=1,order=0),y_axis='mel', fmax=40000,x_axis='time',cmap='gray_r')
        # plt.title('Spectogram')
        # plt.savefig(label+"/spectogram_"+file.replace("/","_"))

def noise_make(sound,numbers_of_samples,factor):
    noise = generator.brown(numbers_of_samples)
    noise = noise*factor
    low = 0.23
    high = 0.8
    b ,a = signal.butter(4,(low,high) , btype="bandpass")  ## filtr pasmowo przepustowy 
    output_signal = signal.filtfilt(b, a, noise)
    #output_signal += sound 
    return output_signal

def saving_spectograms_02(output_signal,sr,ln,path_to_file,label,starting_point,num_of_section):
    path_to_file = "32_7ms_noise/"+label+"/"+path_to_file.replace('/','_')+"spektogram{0}_32".format(starting_point)
    print(output_signal.shape)
    f, t, ps = signal.stft(output_signal, sr, nperseg=ln,noverlap=0,boundary=None)
    t=t+starting_point
    print("max:",np.amax(ps),"min:",np.amin(ps))
    print(ps.shape)
    if(ps.shape==(141,num_of_section)):
        numbers_of_fft_samples = 96
        start_from = 25
        plt.pcolormesh(t, f[start_from:start_from + numbers_of_fft_samples], np.abs(ps[start_from:start_from + numbers_of_fft_samples]),cmap='gray_r')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        cbar = plt.colorbar()
        cbar.set_label("Intensity (dB)")
        plt.savefig(path_to_file+".jpg")
        plt.close()
        np.save(path_to_file,np.abs(ps[start_from:start_from + numbers_of_fft_samples]))

def dividing_to_make_spectograms(sound,cracks,num_of_samples,path_to_file,num_of_section):
    stop = num_of_samples
    start = 0
    cracks=cracks*40
    while(stop<sound.shape[1]):
        is_crack_inside=[]
        print(cracks)
        is_crack_inside = [x > start and x < stop for x in cracks]
        next_crack = [x-stop for x in cracks if x>stop]
        if(any(is_crack_inside)):
            print("tu jest pęknięcie")
            label = "nok"
            offset = 8*num_of_samples/num_of_section
        elif(any([y<num_of_samples for y in next_crack])):
            print("zaraz będzie")
            offset = min(next_crack)+(random.randint(2,8)*num_of_samples/num_of_section)
            label = "ok"
        else:
            print("tu nie ma")
            offset = num_of_samples
            label = "ok"
        print(start,stop)
        print(offset) 
        saving_spectograms_02(sound[0][int(start):int(stop)],
                              40000,
                              int(num_of_samples/num_of_section),
                              path_to_file,
                              label,
                              start/40000,
                              num_of_section
                              )
        noisy_sound = noise_make(sound[0][int(start):int(stop)],num_of_samples,0.006)
        saving_spectograms_02(noisy_sound,
                              40000,
                              int(num_of_samples/num_of_section),
                              path_to_file+"noisy___",
                              label,
                              start/40000,
                              num_of_section
                              )
        start+=offset
        stop+=offset

data_opener = data_prepare()
for folder in ['Pękanie 1','Pękanie 2','Pękanie 3','Pękanie 4']: #,'Pękanie 2','Pękanie 3','Pękanie 4'
    for folder2 in os.listdir(folder):
        cracks = np.array([])
        sound = np.array([])
        for file in os.listdir(folder+'/'+folder2):
            path_to_file = "{0}/{1}/".format(folder,folder2)
            print(path_to_file+file)
            if("_dźwięk.csv" in file):
                sound = np.array(data_opener.open_csv_and_save_to_wav(path=path_to_file,file=file))
                if(sound==[]):
                    break
                sound = sound.reshape(2,-1)
            elif("_pęknięcia_nowe.csv" in file):
                cracks = np.array(data_opener.open_csv_and_count(path=path_to_file,file=file))
        if(len(sound[0])):
            print("Czas zrobic spektogramy")
            print(sound.shape,cracks)
            time_duration = 7 ##ms
            sr=40000 #Hz
            num_of_section = 128/4
            samples_for_one_col = int(0.001*time_duration*sr)
            samples_for_spectogram = int(samples_for_one_col*num_of_section)
            print("spektogram będzie zawierał {0}ms nagrania".format(time_duration*num_of_section))
            print("we need {0} samples for {1} ms. spectogram 128x{3} need {2}".format(samples_for_one_col,time_duration,samples_for_spectogram,num_of_section))
            low = 0.23
            high = 0.8
            b ,a = signal.butter(4,(low,high) , btype="bandpass")  ## filtr pasmowo przepustowy 
            output_signal = signal.filtfilt(b, a, sound)
            #librosa.output.write_wav(path_to_file+'odszumiony{}_{}.wav'.format(low,high),np.array(output_signal[0],dtype='float'),40000,norm=True)
            dividing_to_make_spectograms(output_signal,cracks,samples_for_spectogram,path_to_file,num_of_section)
        else:
            print("Brak dzwięku!")
print(data_opener.number_of_cracks)