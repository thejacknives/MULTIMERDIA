# -*- coding: utf-8 -*-


import librosa
import librosa.display
import sounddevice as sd  
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sc


def normaliza_features(features):
    min_por_coluna = np.amin(features, axis=0)
    max_por_coluna = np.amax(features, axis=0)
    return (features - min_por_coluna) / (max_por_coluna - min_por_coluna)


##Transformar as feaures em stats
def stats(features,tempo,numSong,statsTotal):
    stats = np.zeros(190)
    i = 0
    for feature in features:
        stats[i*7+0]=sc.mean(feature)
        #print("media")
        stats[i*7+1]=np.std(feature)
        #print("desvio")
        stats[i*7+2]=sc.stats.skew(feature)
        #print("assimetria")
        stats[i*7+3]=sc.stats.kurtosis(feature)
        #print("curtose")
        stats[i*7+4]=np.median(feature)
        #print("mediana")
        stats[i*7+5]=np.max(feature)
        #print("max")
        stats[i*7+6]=np.min(feature)
        #print("min")
        i+=1
    #normaliza_features()#rever os max e os mins TODO
    stats[189]=tempo
    statsTotal[numSong]=stats
    
def main():
    if os.path.isfile('./Features/top100_features.csv'):
        # EX 2.1.1
        #features = np.genfromtxt('./Features/top100_features.csv', delimiter=',', skip_header=1)
        #features = features[:, 1:-1]#tirar o nome e o quadrante da song
        # EX 2.1.2
        #features_normalizadas = normaliza_features(features)
        # EX 2.1.3
        #np.savetxt("features_normalizadas.csv", features_normalizadas, delimiter=",")#guarda as features normalizadas
        # --- Load 
        path_to_directory = "Dataset/Musics"
        statsTotal = np.zeros((4,190))
        #fName = "./Queries/MT0000202045.mp3"
        features_total = list()
        warnings.filterwarnings("ignore")
        sung = -1
        for fName in os.listdir(path_to_directory):
            sung+=1
            print("a tratar de", fName, sung)
            y, fs = librosa.load("Dataset/Musics/"+fName, mono=True)#y freq em cada posicao
            features = list()
            mfcc = librosa.feature.mfcc(y=y,n_mfcc=13)#perceber o tamanho da frame e como fazemos os safe
            for i in mfcc:
                features.append(i)
            #add das features dos mfcc
            spec_centroid = librosa.feature.spectral_centroid(y=y).flatten()
            features.append(spec_centroid)
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y).flatten()
            features.append(spec_bandwidth)
            spec_contrast = librosa.feature.spectral_contrast(y=y,n_bands=6)
            for i in spec_contrast:
                features.append(i)
            #dar add as 7 bandas do spectral contrast
            spec_flatness = librosa.feature.spectral_flatness(y=y).flatten()
            features.append(spec_flatness)
            spec_rolloff = librosa.feature.spectral_rolloff(y=y).flatten()
            features.append(spec_rolloff)
            f0 = librosa.core.yin(y=y,fmin=20,fmax=22050/2).flatten()
            features.append(f0)
            rms = librosa.feature.rms(y=y).flatten() #utiliza os valores default do enunciado
            features.append(rms)
            zero_cross_rate = librosa.feature.zero_crossing_rate(y=y).flatten()
            features.append(zero_cross_rate)
            tempo, garbage3 = librosa.beat.beat_track(y=y)
            stats(features,tempo,sung,statsTotal)
        statsTotal = normaliza_features(statsTotal)
        np.savetxt("features_900_smurfs.csv", statsTotal, delimiter=",")
    else:
        print("ficheiro não encontrado.")


if __name__ == "__main__":
    main()

"""
    #--- Load file
    #fName = "--/Queries/MT0000414517.mp3"
    fName = "./Queries/MT0000202045.mp3"
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono = mono)
    print(y.shape)
    print(fs)

    #--- Play Sound
    sd.play(y, sr, blocking=False)

    #--- Plot sound waveform
    plt.figure()
    librosa.display.waveshow(y)

    #--- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    #--- Extract features    
    rms = librosa.feature.rms(y = y)
    rms = rms[0, :]
    print(rms.shape)
    times = librosa.times_like(rms)
    plt.figure(), plt.plot(times, rms)
    plt.xlabel('Time (s)')
    plt.title('RMS')
"""
