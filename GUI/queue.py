import tkinter
import time
import threading
import random
import queue
import pyaudio
import wave
import numpy as np
from my_class.audioClass import audioClass
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import style
import matplotlib
import matplotlib.pyplot as pltlib
import matplotlib.animation as animation
from my_class.network import classification
matplotlib.use("TkAgg")

LARGE_FONTS=("verdana",12)
style.use("ggplot")

# f = Figure(figsize=(5,5), dpi=100)
# a = f.add_subplot(111)

class GuiPart():
    def __init__(self, master, queue, endCommand, resetCommand,stopStartCommand):
        self.network = classification()
        self.network.set_max_min( 0.0006509951221093927,9.303502839404124e-10)
        self.i=0
        self.queue = queue
        # Set up the GUI
        self.continuePlotting=False
        # Add more GUI stuff here depending on your specific needs
        
        master.wm_title("Acoustic analise")
        self.container = tkinter.Frame(master)
        self.container.pack(side="top", fill="both", expand=True)

        self.container.grid_rowconfigure(0,weight=1)
        self.container.grid_columnconfigure(0,weight=1)

        # self.btn_text = tkinter.StringVar(master,"Start")
        # self.btn_text.set("Start")

        button1 = tkinter.Button(master, text="Koniec", command=endCommand)
        button1.pack(side=tkinter.LEFT)

        self.button2 = tkinter.Button(master, text="start", background="green", command=stopStartCommand)
        self.button2.pack(side=tkinter.LEFT) 

        self.e1 = tkinter.Entry(master)
        self.e1.pack(side=tkinter.LEFT)

        button3 = tkinter.Button(master, text='Reset', command=resetCommand)
        button3.pack(side=tkinter.LEFT)

        self.frames  = {}

        frame = StartPage(self.container, endCommand, self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)
        
    def reset(self):
        prefix = self.e1.get()
        self.frames[StartPage].resetData()
        self.frames[StartPage].plotter()
        self.i=0
        return prefix

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def processIncoming(self):
        """Handle all messages currently in the queue, if any."""
        while self.queue.qsize(  ):
            try:
                msg = self.queue.get(0)
                # Check contents of message and do whatever is needed. As a
                # simple test, print it (in real life, you would
                # suitably update the GUI's display in a richer fashion).
                if(self.getContinuePlotting()):
                    prediction = self.network.get_preditction(msg)
                    print(prediction)
                    self.frames[StartPage].addData(prediction)
                    if(self.i>5):
                        self.frames[StartPage].plotter()
                        self.i=0
                    else:
                        self.i+=1
                print(msg)
            except self.queue.Empty:
                # just on general principles, although we don't
                # expect this branch to be taken in this case
                pass

    def change_state(self):
        print("button pushed")
        if self.continuePlotting == True:
            self.button2["text"]="Start"
            self.button2["background"]="green"
            self.continuePlotting = False
        else:
            self.button2["text"]="Stop"
            self.button2["background"]="red"
            self.continuePlotting = True
            self.reset()

    def getContinuePlotting(self):
        return self.continuePlotting

class StartPage(tkinter.Frame):
    def __init__(self, parent,endCommand, controler):
        tkinter.Frame.__init__(self, parent)
        label = tkinter.Label(self,text="System akustycznego monitorowania napawania laserowego wersja 1.0", font=LARGE_FONTS)
        label.pack(pady=10, padx=10)
        self.plotData=[]
        self.plotTime=[]
        
        fig = Figure()
    
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.grid()

        self.canvas = FigureCanvasTkAgg(fig,self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(self.canvas,self)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        
    def addData(self,msg):
        if(len(self.plotData)>250):
            self.plotData.pop(0)
            self.plotData.append(msg)
            self.plotTime=list(map(lambda x: (x+0.224), self.plotTime))
        else:
            self.plotData.append(msg)
            self.plotTime.append(len(self.plotTime)*0.224)
    
    def resetData(self):
        self.plotData.clear()
        self.plotTime.clear()

    def plotter(self):
        self.ax.cla()
        self.ax.grid()
        #self.ax.set_yscale('log')
        self.ax.autoscale()
        self.ax.plot(self.plotTime,self.plotData, marker='o', color='orange')
        self.canvas.draw()

    # def gui_handler(self,msg):
    #     threading.Thread(target=self.plotter(msg)).start()

class ThreadedClient:
    """
    Launch the main part of the GUI and the worker thread. periodicCall and
    endApplication could reside in the GUI part, but putting them here
    means that you have all the thread controls in a single place.
    """
    def __init__(self, master):
        """
        Start the GUI and the asynchronous threads. We are in the main
        (original) thread of the application, which will later be used by
        the GUI as well. We spawn a new thread for the worker (I/O).
        """
        

        self.audio = audioClass()

        self.master = master

        # Create the queue
        self.queue = queue.Queue()

        # Set up the GUI part
        self.gui = GuiPart(master, self.queue, self.endApplication,self.reset,self.stop_start)

        # Set up the thread to do asynchronous I/O
        # More threads can also be created and used, if necessary
        self.running = 1
        self.thread1 = threading.Thread(target=self.workerThread1)
        self.thread1.start(  )

        # Start the periodic call in the GUI to check if the queue contains
        # anything
        self.periodicCall(  )

    def periodicCall(self):
        """
        Check every 300 ms if there is something new in the queue.
        """
        self.gui.processIncoming()
        if not self.running:
            # This is the brutal stop of the system. You may want to do
            # some cleanup before actually shutting it down.
            self.audio.end()
            import sys
            sys.exit(0)
            print("exit")
        self.master.after(300, self.periodicCall)

    def workerThread1(self):
        """
        This is where we handle the asynchronous I/O. For example, it may be
        a 'select(  )'. One important thing to remember is that the thread has
        to yield control pretty regularly, by select or otherwise.
        """
        while self.running:
            # To simulate asynchronous I/O, we create a random number at
            # random intervals. Replace the following two lines with the real
            # thing.
            waveData = self.audio.getwave()
            self.audio.isFull()
            msg = waveData
            self.queue.put(msg)

    def endApplication(self):
        self.running = 0

    def reset(self):
        prefix = self.gui.reset()
        self.audio.reset(prefix)
        print("reset")

    def stop_start(self):
        self.gui.change_state()
        print("stop")

root = tkinter.Tk(  )
client = ThreadedClient(root)
root.mainloop( )