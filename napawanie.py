import pandas as pd
import csv 
import numpy as np
import librosa
import os


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
                        array.append(probe.replace(',','.'))
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

data_opener = data_prepare()
for folder in ['Pękanie 3']: #,'Pękanie 2','Pękanie 3','Pękanie 4'
    for folder2 in os.listdir(folder):
        cracks = np.array([])
        sound = np.array([])
        for file in os.listdir(folder+'/'+folder2):
            path_to_file = "{0}/{1}/".format(folder,folder2)
            print(path_to_file+file)
            if("6_dźwięk.csv" in file):
                sound = np.array(data_opener.open_csv_and_save_to_wav(path=path_to_file,file=file))
                sound = sound.reshape(2,-1)
            elif("6_pęknięcia.csv" in file):
                cracks = np.array(data_opener.open_csv_and_count(path=path_to_file,file=file))
        if(len(sound)):
            print("Czas zrobic spektogramy")
            print(sound.shape,cracks)
            time_duration = 10 ##ms
            sr=40000 #Hz
            samples_for_one_col = 0.001*time_duration*sr
            samples_for_spectogram = int(samples_for_one_col*128)
            print("we need {0} samples for {1} ms. spectogram 128x128 need {2}".format(samples_for_one_col,time_duration,samples_for_spectogram))
        else:
            print("Brak dzwięku!")
print(data_opener.number_of_cracks)