import pyaudio
import wave
import numpy as np
import os
class audioClass:
    def __init__(self):
        self.TIME_DURATION = 7 ##ms
        self.SAMPLE_RATE = 40000 #Hz
        self.NUM_OF_COLS = 32
        self.SAMPLES_FOR_COL = int(0.001*self.TIME_DURATION*self.SAMPLE_RATE)
        self.SAMPLES_FOR_ONE_SPECTROGRAM = int(self.SAMPLES_FOR_COL*self.NUM_OF_COLS)
        self.FORMAT = pyaudio.paInt16 # We use 16bit format per sample
        self.CHANNELS = 1
        self.RATE = 40000
        self.CHUNK = self.SAMPLES_FOR_ONE_SPECTROGRAM # 1024bytes of data red from a buffer # liczba próbek potrzebna do spektogramu.
        self.WAVE_OUTPUT_FILENAME = "napawanie.wav"
        self.AUDIO = pyaudio.PyAudio()
        self.i=0.0
        self.fileNumber=0.0
        self.prefix = ""

        self.stream = self.AUDIO.open(format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True)#,
                    #frames_per_buffer=CHUNK)
        self.stream.start_stream()
        self.frames = []

    def save(self):
        file = "nagrania/Powloka_{0}/napawanie{1}.wav".format(str(self.prefix),self.fileNumber)
        self.ensure_dir(file)
        WAVE_OUTPUT_FILENAME = file
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.AUDIO.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
        self.frames=[]

    def ensure_dir(self,file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def end(self):
        self.stream.stop_stream()
        self.stream.close()
        self.AUDIO.terminate()
        self.save()

    def getwave(self):
        data = self.stream.read(self.CHUNK)
        waveData = np.array(wave.struct.unpack("%dh"%(self.CHUNK), data))
        self.frames.append(data)
        self.i+=(self.TIME_DURATION*32/1000)
        return waveData

    def isFull(self):
        if(self.i>10):
            self.save()
            self.fileNumber += self.i
            self.i = 0.0

    def reset(self,prefix):
        self.save()
        self.prefix = prefix
        self.fileNumber = 0
        self.i = 0
