import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import csv 
import numpy as np
import os
import keras


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

    def open_csv_and_count(self,path,file):
        with open(path+file, 'r',newline='\n',encoding="Latin-1") as csvfile:
            sound = csv.reader(csvfile, delimiter='\t')
            num_of_cracks = 0
            try:
                array = []
                for channel in sound:
                    for probe in channel:
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
        librosa.display.specshow(keras.utils.normalize(ps,axis=-1,order=2),y_axis='linear', fmax=40000,x_axis='time',cmap='gray_r')
        plt.title('Spectrogram')
        plt.savefig(label+"/spectogram_"+file.replace("/","_"))

def dividing_to_make_spectograms(sound,cracks,num_of_samples,file):
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
            offset = 32*num_of_samples/128
        elif(any([y<num_of_samples for y in next_crack])):
            print("zaraz będzie")
            offset = min(next_crack)+(num_of_samples/32)
            label = "ok"
        else:
            print("tu nie ma")
            offset = num_of_samples
            label = "ok"
        print(start,stop)
        print(offset)
        start+=offset
        stop+=offset 
        file_name=file+str(int(start))
        saving_spectograms(sound[0][int(start):int(stop)],40000,int(num_of_samples/128),file_name,label)

data_opener = data_prepare()
for folder in ['Pękanie 4']: #,'Pękanie 2','Pękanie 3','Pękanie 4'
    for folder2 in os.listdir(folder):
        cracks = np.array([])
        sound = np.array([])
        for file in os.listdir(folder+'/'+folder2):
            path_to_file = "{0}/{1}/".format(folder,folder2)
            print(path_to_file+file)
            if("5_dźwięk.csv" in file):
                sound = np.array(data_opener.open_csv_and_save_to_wav(path=path_to_file,file=file))
                sound = sound.reshape(2,-1)
            elif("5_pęknięcia.csv" in file):
                cracks = np.array(data_opener.open_csv_and_count(path=path_to_file,file=file))
        if(len(sound)):
            print("Czas zrobic spektogramy")
            print(sound.shape,cracks)
            time_duration = 10 ##ms
            sr=40000 #Hz
            samples_for_one_col = int(0.001*time_duration*sr)
            samples_for_spectogram = int(samples_for_one_col*128)
            print("spektogram będzie zawierał {0}ms nagrania".format(time_duration*128))
            print("we need {0} samples for {1} ms. spectogram 128x128 need {2}".format(samples_for_one_col,time_duration,samples_for_spectogram))
            dividing_to_make_spectograms(sound,cracks,samples_for_spectogram,path_to_file)
        else:
            print("Brak dzwięku!")
print(data_opener.number_of_cracks)