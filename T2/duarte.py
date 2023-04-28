#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
    Duarte Meneses  2019216949
    Inês Marçal     2019215917
    Patricia Costa  2019213995
"""

import librosa #https://librosa.org/
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import spatial

test = False

def normalize(arrToNorm):
    normalized = np.zeros(arrToNorm.shape)
    
    for i in range(len(arrToNorm[0])):
        max_v = arrToNorm[:, i].max()
        min_v = arrToNorm[:, i].min()
        if (max_v == min_v):
            normalized[:, i] = 0
        else:
            normalized[:, i] = (arrToNorm[:, i] - min_v)/(max_v - min_v)
        
    return normalized

def statistics(feature):
    mean = np.mean(feature)
    desv = np.std(feature)
    skew = st.skew(feature)
    kurto = st.kurtosis(feature)
    median = np.median(feature)
    max_m = feature.max()
    min_m = feature.min() 
    
    return np.array([mean, desv, skew, kurto, median, max_m, min_m])

def pergunta2():
    # 2.1 - Processar as features do ficheiro top100_features.csv
    
    #2.1.1 - Ler o ficheiro e criar um array numpy com as features disponibilizadas.
    top100_features = np.genfromtxt("Features - Audio MER\\top100_features.csv", dtype = np.str, delimiter=",")
    fNames = top100_features[1:, 0]
    top100_features = top100_features[1::, 1:(len(top100_features[0])-1)].astype(np.float)
    
    if test:
        print(top100_features)
    
    #2.1.2 - Normalizar as features no intervalo [0, 1]
    
    top100_featuresN = normalize(top100_features)
        
    if test:
        print(top100_featuresN)
   
    
    #2.1.3 - Criar e gravar em ficheiro um array numpy com as features extraídas
    #linhas = músicas
    #colunas = valores das features
    
    np.savetxt("Features - Results\\top100_features_normalized.csv", top100_featuresN, fmt = "%lf", delimiter= ",")
    
    #2.2 - Extrair features da framework librosa
    #2.2.1 - Para os 900 ficheiros da BD, extrair as seguintes features:
    #--- Load file
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    features= np.arange(9000, dtype=object).reshape((900,10))
    stat = np.zeros((900,190), dtype=np.float64)
    
    line = 0
    
    for name in fNames:
        name = name.replace("\"", "")
        fName = "MER_audio_taffc_dataset\\music\\" + name + ".mp3"
        print("Song number: ", line+1, "song: ", name)
        
        y, fs = librosa.load(fName, sr=sr, mono = mono)
        
        #Features Espectrais
        #2.2.2 - Calcular as 7 estatísticas típicas sobre as features anteriores
        
        mfcc = librosa.feature.mfcc(y=y,n_mfcc=13)
        features[line][0] = mfcc
        for i in range(mfcc.shape[0]):
            stat[line, i*7 : i*7+7] = statistics(mfcc[i, :])
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y)[0,:]
        features[line][1] = spectral_centroid
        stat[line, 91 : 91+7] = statistics(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)[0,:]
        features[line][2] = spectral_bandwidth
        stat[line, 98 : 98+7] = statistics(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y)
        features[line][3] = spectral_contrast
        for i in range(spectral_contrast.shape[0]):
            stat[line, 105+i*7 : 105+(i*7+7)] = statistics(spectral_contrast[i, :])
        
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0,:]
        features[line][4] = spectral_flatness
        stat[line, 154 : 154+7] = statistics(spectral_flatness)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y)[0,:]
        features[line][5] = spectral_rolloff
        stat[line, 161 : 161+7] = statistics(spectral_rolloff)
        
        #Features Temporais
        f0 = librosa.yin(y=y, fmin=20, fmax=fs/2)
        f0[f0==fs/2]=0
        features[line][6] = f0
        stat[line, 168 : 168+7] = statistics(f0)
        
        rms = librosa.feature.rms(y=y)[0,:]
        features[line][7] = rms
        stat[line, 175 : 175+7] = statistics(rms)
        
        zero_cross = librosa.feature.zero_crossing_rate(y=y)[0,:]
        features[line][8] = zero_cross
        stat[line, 182 : 182+7] = statistics(zero_cross)
        
        #Outras features
        time = librosa.beat.tempo(y=y)
        stat[line, 189] = features[line][9] = time[0]
    
        line += 1

    if test:
        print(stat)
    
    statisticN = normalize(stat)
    
    np.savetxt("Features - Results\\features_statistics.csv", statisticN, fmt = "%lf", delimiter= ",");
    
    if test:
        print(statisticN)
        
        
def pergunta3():
    #3 - Implementação de métricas de similaridade.
    #3.1 - Desenvolver o código Python/numpy para calcular as seguintes métricas de similaridade:
    
    top100_features = np.genfromtxt("Features - Results\\top100_features_normalized.csv", dtype = np.float, delimiter=",")

    statistic = np.genfromtxt("Features - Results\\features_statistics.csv", dtype = np.float, delimiter=",")
   
    euclidiana = np.zeros((900,900))
    manhattan = np.zeros((900,900))
    cosseno = np.zeros((900,900))
    
    euclidianaF = np.zeros((900,900))
    manhattanF = np.zeros((900,900))
    cossenoF = np.zeros((900,900))
    
    
    for j in range(900):
        print(j)
        for i in range(900):
         
            euclidianaF[j][i] = np.linalg.norm(statistic[j]- statistic[i])
            manhattanF[j][i] = spatial.distance.cityblock(statistic[j], statistic[i])
            cossenoF[j][i] = spatial.distance.cosine(statistic[j], statistic[i])
           
        
            euclidiana[j][i] = np.linalg.norm(top100_features[j]- top100_features[i])
            manhattan[j][i] = spatial.distance.cityblock(top100_features[j], top100_features[i])
           
            cosseno[j][i] = spatial.distance.cosine(top100_features[j], top100_features[i])


    #3.2. Criar e gravar em ficheiro 6 matrizes de similaridade (900x900)
  
    #Distância Euclidiana
    np.savetxt("Features - Results\\top100_euclidiana.csv", euclidiana, fmt = "%lf", delimiter= ",")
    np.savetxt("Features - Results\\features_euclidiana.csv", euclidianaF, fmt = "%lf", delimiter= ",")

    #Distância de Manhattan
    np.savetxt("Features - Results\\top100_manhattan.csv", manhattan, fmt = "%lf", delimiter= ",")
    np.savetxt("Features - Results\\features_manhattan.csv", manhattanF, fmt = "%lf", delimiter= ",")
 
    #Distância do Coseno    
    np.savetxt("Features - Results\\top100_cosseno.csv", cosseno, fmt = "%lf", delimiter= ",")
    np.savetxt("Features - Results\\features_cosseno.csv", cossenoF, fmt = "%lf", delimiter= ",")
  
def rank_simi():
    #3.3. Criar rankings de similaridade
    
    euclidianaFile100 = np.genfromtxt("Features - Results\\top100_euclidiana.csv", dtype = np.float, delimiter=",")
    manhattanFile100 = np.genfromtxt("Features - Results\\top100_manhattan.csv", dtype = np.float, delimiter=",")
    cossenoFile100 = np.genfromtxt("Features - Results\\top100_cosseno.csv", dtype = np.float, delimiter=",")
    
    euclidianaFileF = np.genfromtxt("Features - Results\\features_euclidiana.csv", dtype = np.float, delimiter=",")
    manhattanFileF = np.genfromtxt("Features - Results\\features_manhattan.csv", dtype = np.float, delimiter=",")
    cossenoFileF = np.genfromtxt("Features - Results\\features_cosseno.csv", dtype = np.float, delimiter=",")
    
    
    top100_features = np.genfromtxt("Features - Audio MER\\top100_features.csv", dtype = np.str, delimiter=",")
    fNames = top100_features[1:, 0]
    for i in range(len(fNames)):
        fNames[i] = fNames[i].replace("\"", "")
      
    
    print('MT0000202045\n')
    rank_e1001, rank_m1001, rank_c1001, rank_ef1, rank_mf1, rank_cf1 = ranking_3('MT0000202045', fNames, euclidianaFile100, manhattanFile100, cossenoFile100, euclidianaFileF, manhattanFileF, cossenoFileF)
    print("-------------------------")
    
    print('MT0000379144\n')
    rank_e1002, rank_m1002, rank_c1002, rank_ef2, rank_mf2, rank_cf2 = ranking_3('MT0000379144', fNames, euclidianaFile100, manhattanFile100, cossenoFile100, euclidianaFileF, manhattanFileF, cossenoFileF)
    print("-------------------------")
    
    print('MT0000414517\n')
    rank_e1003, rank_m1003, rank_c1003, rank_ef3, rank_mf3, rank_cf3 = ranking_3('MT0000414517', fNames, euclidianaFile100, manhattanFile100, cossenoFile100, euclidianaFileF, manhattanFileF, cossenoFileF)
    print("-------------------------")
    
    print('MT0000956340\n')
    rank_e1004, rank_m1004, rank_c1004, rank_ef4, rank_mf4, rank_cf4 = ranking_3('MT0000956340', fNames, euclidianaFile100, manhattanFile100, cossenoFile100, euclidianaFileF, manhattanFileF, cossenoFileF)
    print("-------------------------")
    
    pergunta4()
    
    metadados_mat = np.genfromtxt("MER_audio_taffc_dataset\\similaridade.csv", dtype = np.int, delimiter=",")
    
    print('MT0000202045\n')
    rank1 = ranking_4('MT0000202045', fNames, metadados_mat)
    print("-------------------------")
    
    print('MT0000379144\n')
    rank2 = ranking_4('MT0000379144', fNames, metadados_mat)
    print("-------------------------")
    
    print('MT0000414517\n')
    rank3 = ranking_4('MT0000414517', fNames, metadados_mat)
    print("-------------------------")
    
    print('MT0000956340\n')
    rank4 = ranking_4('MT0000956340', fNames, metadados_mat)
    print("-------------------------")
    
    
    print('\nMT0000202045')
    metric_precision(rank_e1001, rank_m1001, rank_c1001, rank_ef1, rank_mf1, rank_cf1, rank1)
    print('\nMT0000379144')
    metric_precision(rank_e1002, rank_m1002, rank_c1002, rank_ef2, rank_mf2, rank_cf2, rank2)
    print('\nMT0000414517')
    metric_precision(rank_e1003, rank_m1003, rank_c1003, rank_ef3, rank_mf3, rank_cf3, rank3)
    print('\nMT0000956340')
    metric_precision(rank_e1004, rank_m1004, rank_c1004, rank_ef4, rank_mf4, rank_cf4, rank4)
    
    
    
    
def ranking_3(musictoanalyse, fNames, euclidianaFile100, manhattanFile100, cossenoFile100, euclidianaFileF, manhattanFileF, cossenoFileF):
    music_pos = np.where(fNames == musictoanalyse)[0][0]

    rank_e100 = np.argsort(euclidianaFile100[music_pos, :])
    rank_e100 = rank_e100[1:21]
    
    print("RANKING EUCLIDIANA 100 FEATURES")
    for pos in rank_e100:
        print(fNames[pos] + ".mp3", end = " , ")
    print(end = "\n\n")
        
    rank_m100 = np.argsort(manhattanFile100[music_pos, :])
    rank_m100 = rank_m100[1:21]
    
    print("RANKING MANHATTAN 100 FEATURES")
    for pos in rank_m100:
        print(fNames[pos] + ".mp3", end = " , ")
    print(end = "\n\n")
    
    rank_c100 = np.argsort(cossenoFile100[music_pos, :])
    rank_c100 = rank_c100[1:21]
    
    print("RANKING COSSENO 100 FEATURES")
    for pos in rank_c100:
        print(fNames[pos] + ".mp3", end = " , ")
    print(end = "\n\n")
    
    rank_ef = np.argsort(euclidianaFileF[music_pos, :])
    rank_ef = rank_ef[1:21]
    
    print("RANKING EUCLIDIANA FEATURES")
    for pos in rank_ef:
        print(fNames[pos] + ".mp3", end = " , ")
    print(end = "\n\n")
    
    rank_mf = np.argsort(manhattanFileF[music_pos, :])
    rank_mf = rank_mf[1:21]
    
    print("RANKING MANHATTAN FEATURES")
    for pos in rank_mf:
        print(fNames[pos] + ".mp3", end = " , ")
    print(end = "\n\n")
    
    rank_cf = np.argsort(cossenoFileF[music_pos, :])
    rank_cf = rank_cf[1:21]
    
    print("RANKING COSSENO FEATURES")
    for pos in rank_cf:
        print(fNames[pos] + ".mp3", end = " , ")
    print(end = "\n\n")
    
    return rank_e100, rank_m100, rank_c100, rank_ef, rank_mf, rank_cf
    
    

def pergunta4():
    metadados = np.genfromtxt("MER_audio_taffc_dataset\\panda_dataset_taffc_metadata.csv", dtype = np.str, delimiter=",")[1::, :]
    
    meta_mat = np.zeros((900,900))
    
      
    for i in range(900):
        for j in range(900):
            res = 0
            if (metadados[i][1].replace("\"", "") == metadados[j][1].replace("\"", "")):
                res += 1
            if (metadados[i][3].replace("\"", "") == metadados[j][3].replace("\"", "")):
                res += 1 
            aux_1 = metadados[i][9].replace("\"", "").split("; ")
            aux_2 = metadados[j][9].replace("\"", "").split("; ")     
            for k in range(len(aux_1)):
                if (aux_1[k] in aux_2):
                    res += 1
                    
            aux_1 = metadados[i][11].replace("\"", "").split("; ")
            aux_2 = metadados[j][11].replace("\"", "").split("; ")
            
            for k in range(len(aux_1)):
                if (aux_1[k] in aux_2):
                    res += 1
            
            meta_mat[i][j] = res
      
    np.savetxt("MER_audio_taffc_dataset\\similaridade.csv", meta_mat, fmt = "%d", delimiter= ",")
    

    
def ranking_4(musictoanalyse, fNames, metadados_mat):
    music_pos = np.where(fNames == musictoanalyse)[0][0]

    rank = metadados_mat[music_pos, :].argsort()[::-1]
    
    print("RANKING METADADOS")
    
    rank_ = np.zeros(20)
    
    num = 0
    for pos in rank:
        if (fNames[pos] != musictoanalyse):
            print(fNames[pos] + ".mp3", end = " , ")
            rank_[num] = pos
            num += 1
        if num == 20:
            break
            
    print(end = "\n\n")
    
    return rank_


def metric_precision(rank_e100, rank_m100, rank_c100, rank_ef, rank_mf, rank_cf, rank):
    
    de = np.zeros(6)
    
    res = 0
    for r in rank_e100:
        if r in rank:
            res += 1
            
    de[0] = (res/20) * 100
    #print("SIMILARIDADE EUCLIDIANA 100 FEATURES:", (res/20) * 100)
    
    res = 0
    for r in rank_m100:
        if r in rank:
            res += 1
            
    de[1] = (res/20) * 100
    #print("SIMILARIDADE MANHATTAN 100 FEATURES:", (res/20) * 100)
    
    res = 0
    for r in rank_c100:
        if r in rank:
            res += 1
            
    de[2] = (res/20) * 100
    #print("SIMILARIDADE COSSENO 100 FEATURES:", (res/20) * 100)
    
    res = 0
    for r in rank_ef:
        if r in rank:
            res += 1
    
    de[3] = (res/20) * 100
    #print("SIMILARIDADE EUCLIDIANA FEATURES:", (res/20) * 100)
    
    res = 0
    for r in rank_mf:
        if r in rank:
            res += 1
            
    de[4] = (res/20) * 100
    #print("SIMILARIDADE MANHATTAN FEATURES:", (res/20) * 100) 
   
    res = 0
    for r in rank_cf:
        if r in rank:
            res += 1
            
    de[5] = (res/20) * 100
    #print("SIMILARIDADE COSSENO FEATURES:", res/20) 
    
    print(de)
    
    
    
if __name__ == "__main__":   
    pergunta2()
    
    pergunta3()
        
    rank_simi()