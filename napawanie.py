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
            #print(sound.line_num)
            channel_num = 0
            try:
                for channel in sound:
                    array = []
                    for probe in channel:
                        array.append(probe.replace(',','.'))
                    print(len(array))
                    librosa.output.write_wav(path+file.replace("_dźwięk.csv","_nagranie.wav"),np.array(array,dtype='float'),40000,norm=True)
                    channel_num+=1

                print("loaded sound with {0} chanels".format(channel_num))
            except csv.Error as e:
                print("Can't load",'path %s, line %d: %s' % (path, sound.line_num, e))

    def open_csv_and_count(self,path,file):
        with open(path+file, 'r',newline='\n',encoding="Latin-1") as csvfile:
            sound = csv.reader(csvfile, delimiter='\t')
            print(sound.line_num)
            num_of_cracks = 0
            try:
                array = []
                for channel in sound:
                    for probe in channel:
                        array.append(float(probe.replace(',','.')))
                        num_of_cracks+=1
                    print(array)
                self.number_of_cracks+=num_of_cracks
                print("loaded sound with {0} chanels".format(num_of_cracks))
            except csv.Error as e:
                print("Can't load",'path %s, line %d: %s' % (path, sound.line_num, e))

data_opener = data_prepare()
for folder in ['Pękanie 1','Pękanie 2','Pękanie 3','Pękanie 4']:
    for folder2 in os.listdir(folder):
        for file in os.listdir(folder+'/'+folder2):
            path_to_file = "{0}/{1}/".format(folder,folder2)
            print(path_to_file+file)
            if("_dźwięk.csv" in file):
                data_opener.open_csv_and_save_to_wav(path=path_to_file,file=file)
            elif("_pęknięcia.csv" in file):
                data_opener.open_csv_and_count(path=path_to_file,file=file)
print(data_opener.number_of_cracks)