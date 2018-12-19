import matplotlib
import matplotlib.animation as animation
import tkinter as tk
import pyaudio
import wave
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import style
from tkinter import ttk
from my_class.network import classification
matplotlib.use("TkAgg")

TIME_DURATION = 7 ##ms
SAMPLE_RATE = 40000 #Hz
NUM_OF_COLS = 32
SAMPLES_FOR_COL = int(0.001*TIME_DURATION*SAMPLE_RATE)
SAMPLES_FOR_ONE_SPECTROGRAM = int(SAMPLES_FOR_COL*NUM_OF_COLS)

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 1
RATE = 40000
CHUNK = SAMPLES_FOR_ONE_SPECTROGRAM # 1024bytes of data red from a buffer # liczba próbek potrzebna do spektogramu.
WAVE_OUTPUT_FILENAME = "file.wav"

AUDIO = pyaudio.PyAudio()

LARGE_FONTS=("verdana",12)
style.use("ggplot")

f = Figure(figsize=(5,5), dpi=100)
a = f.add_subplot(111)

stream = AUDIO.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True)#,
                    #frames_per_buffer=CHUNK)
plot_data = []
plot_time = []

network = classification()
network.set_max_min( 0.0006509951221093927,9.303502839404124e-10)
frames=[]

def animate(i):
    stream.start_stream()
    data = stream.read(CHUNK)
    waveData = np.array(wave.struct.unpack("%dh"%(CHUNK), data))
    spektrogram = network.get_preditction(waveData)
    a.clear()
    a.pcolormesh(spektrogram,cmap='gray_r')
    
    
    if(i>int(RATE / CHUNK * 7)):
        stream.stop_stream()
        stream.close()
        AUDIO.terminate()
        WAVE_OUTPUT_FILENAME = "nagranie1.wav"
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(AUDIO.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        exit()
    frames.append(data)
    # stop Recording
    

    # plot_data.append(max(waveData))
    # last=0
    # if(plot_time):
    #     last = plot_time[-1]
    # plot_time.append(TIME_DURATION*NUM_OF_COLS+last)
    # a.set_ylim(min(plot_data),max(plot_data))
    # a.clear()
    # a.plot(plot_time,plot_data)


class weldingApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Acoustic analise")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)

        self.frames  = {}

        for F in (StartPage,):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):
    def __init__(self, parent, controler):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self,text="Page 1", font=LARGE_FONTS)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Visit Start Page",
                            command=lambda: controler.show_frame(StartPage))
        button1.pack()   

        canvas = FigureCanvasTkAgg(f,self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas,self)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
if __name__ == "__main__":
    app=weldingApp()
    ani = animation.FuncAnimation(f,animate)
    app.mainloop()