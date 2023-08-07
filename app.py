import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import io
import pickle

def download_model_weights(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Download the model weights if not already present
model_url = "https://github.com/omagarwal2002/Smart-Speech-Analysis-Minor1/raw/main/SER_by_NOR.pkl"
local_model= "SER_by_NOR.pkl"
download_model_weights(model_url, local_model)

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def extract_feature(y, mfcc, chroma, mel, sample_rate):
    if chroma:
        stft = np.abs(librosa.stft(y))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sample_rate).T, axis=0)  
        result = np.hstack((result, mel))
    return result

def main():
    st.title("SPEECH EMOTION RECOGNIZER")
    st.write("Upload an audio clip and get its emotion present in it.")

    # Record Audio
    audio_file = st.file_uploader("Upload audio", type=["wav"])

    if st.button("Extract Features and Make Prediction") and audio_file is not None:
        audio_data = audio_file.read()
        y, sample_rate = sf.read(io.BytesIO(audio_data))
        envelope_mask = envelope(y, rate=sample_rate, threshold=0.0005)
        y_filtered = y[envelope_mask]
        ans =[]
        new_feature = extract_feature(y_filtered, mfcc=True, chroma=True, mel=True, sample_rate=sample_rate)
        ans.append(new_feature)
        ans = np.array(ans)

        # Load the pre-trained model (change the filename to your actual model's filename)
        Pkl_Filename = local_model
        with open(Pkl_Filename, 'rb') as file:  
            model = pickle.load(file)

        # Make prediction
        prediction = model.predict(ans)
        emotion = prediction[0]

        # Show prediction result
        st.write("Emotion Present:")
        st.write(emotion)

if __name__ == "__main__":
    main()
