
import sounddevice as sd
import soundfile as sf
from tkinter import *
from PIL import ImageTk, Image
import pyaudio
import wave
import numpy as np
import soundfile
import librosa
import pandas as pd
from scipy.io import wavfile
import pickle
from playsound import playsound
import os
import simpleaudio as sa
import speech_recognition as sr

def envelope(y , rate, threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

def voice_to_text():
    r=sr.Recognizer()
    with sr.AudioFile('F:/VS Code/Minor1_midsem/self recorded/output1.wav') as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(text)
        except:
            print('error')
  
def Play_sound():
    filename = 'F:/VS Code/Minor1_midsem/self recorded/output1.wav'
    #playsound(filename)
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def Voice_rec():
    os.remove(r"F:\VS Code\Minor1_midsem\self recorded\cleaned.wav")
    os.remove(r"F:\VS Code\Minor1_midsem\self recorded\output1.wav")
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 2 
    RATE = 44100 #sample rate
    RECORD_SECONDS = int(duration.get())
    print(RECORD_SECONDS)
    WAVE_OUTPUT_FILENAME = "F:/VS Code/Minor1_midsem/self recorded/output1.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

    print("* recording")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    file = 'F:/VS Code/Minor1_midsem/self recorded/output1.wav'
    signal , rate = librosa.load(file, sr=16000)
    mask = envelope(signal,rate, 0.0005)
    wavfile.write(filename= r'F:\VS Code\Minor1_midsem\self recorded\cleaned.wav', rate=rate,data=signal[mask])
    voice_to_text()

def Make_prediction():
    file1 = 'F:/VS Code/Minor1_midsem/self recorded/cleaned.wav'
    ans =[]
    new_feature = extract_feature(file1, mfcc=True, chroma=True, mel=True)
    ans.append(new_feature)
    ans = np.array(ans)
    Pkl_Filename = "F:/VS Code/Minor1_midsem/SER_by_NOR.pkl"
    with open(Pkl_Filename, 'rb') as file:  
        Speech_Emotion_Recognition = pickle.load(file)
    x=Speech_Emotion_Recognition.predict(ans)
    emotion=x[0]
    Label(root, text="Emotion : "+emotion, font=("Helvetica", 16)).place(x=173, y=330)


root = Tk()

Label(root, text=" Voice Recoder : "
     ).grid(row=0, sticky=W, rowspan=5)
  
root.geometry('500x500')
root.configure(bg='lemon chiffon')
root.resizable(0,0)
root.title("EMOTION RECOGNIZER")

img = ImageTk.PhotoImage(Image.open("F:/VS Code/Minor1_midsem/record.jpeg").resize((500, 500)))

Label( root, image = img).place(x = 00,y = 00)
duration = StringVar()
Label(root, text = 'Enter duration for rec in sec:', font = 'arial 10').place(x= 50 , y = 10)
seconds = Entry(root, width = 23,textvariable = duration).place(x = 300 , y = 10)
b1 = Button(root, text="Start Recording", command=Voice_rec)
b1.place(x=100, y=60)
b2 = Button(root, text="Make Prediction", command=Make_prediction)
b2.place(x=200, y=100)
b3 = Button(root, text="Play Recording", command=Play_sound)
b3.place(x=300, y=60) 

mainloop()