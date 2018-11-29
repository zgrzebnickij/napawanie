import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import csv 
import numpy as np
import os
import random


def open_csv_and_count(path,file,suma):
        with open(path+file, 'r',newline='\n',encoding="Latin-1") as csvfile:
            sound = csv.reader(csvfile, delimiter='\t')
            try:
                array = []
                for channel in sound:
                    for probe in channel:
                        #array.append(float(probe.replace(',','.')))
                        array = list(map(float,[x.replace(',','.') for x in probe.split()]))
                        suma+=len(array)
                print("{1} {0}".format(array,path+file))
                return suma
            except csv.Error as e:
                print("Can't load",'path %s, line %d: %s' % (path, sound.line_num, e))

suma = 0
for folder in ['Pękanie 1','Pękanie 2','Pękanie 3','Pękanie 4']: #,'Pękanie 2','Pękanie 3','Pękanie 4'
    for folder2 in os.listdir(folder):
        cracks = np.array([])
        sound = np.array([])
        for file in os.listdir(folder+'/'+folder2):
            path_to_file = "{0}/{1}/".format(folder,folder2)
            
            if("_pęknięcia_nowe.csv" in file):
                #print(path_to_file+file)
                suma = np.array(open_csv_and_count(path=path_to_file,file=file,suma=suma))
print(suma)