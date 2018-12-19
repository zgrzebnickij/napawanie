try:
    import pyaudio
    import numpy as np
    import pylab
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    import time
    import sys
    import seaborn as sns
    from  keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
    from keras.models import Sequential
    import time
    from scipy import signal
    import wave
except:
    print("Something didn't import")

low = 0.23
high = 0.8
b ,a = signal.butter(4,(low,high) , btype="bandpass")

def making_model(input_shape):

    model = Sequential()

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=(128,128,1)))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
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
    return model

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

model = making_model((128,128))

# Prepare the Plotting Environment with random starting values
time_duration = 7 ##ms
sr=40000 #Hz
num_of_section =128
samples_for_one_col = int(0.001*time_duration*sr)
samples_for_spektrogram = int(samples_for_one_col*num_of_section)

FORMAT = pyaudio.paInt8 # We use 16bit format per sample
CHANNELS = 1
RATE = 40000
CHUNK = samples_for_spektrogram # 1024bytes of data red from a buffer # liczba próbek potrzebna do spektogramu.
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True)#,
                    #frames_per_buffer=CHUNK)

global keep_going
keep_going = True

# Open the connection and start streaming the data
stream.start_stream()
print("\n+---------------------------------+")
print("| Press Ctrl+C to Break Recording |")
print("+---------------------------------+\n")

# Loop so program doesn't end while the stream callback's
# itself for new data
while keep_going:
    try:
        spectrogram = Prepare_spektrogram(stream.read(CHUNK),128,128,samples_for_spektrogram)
        print(model.predict(spektrogram.reshape(1,128,128,1)))
    except KeyboardInterrupt:
        keep_going=False
    except:
        pass

# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
stream.stop_stream()
stream.close()

audio.terminate()