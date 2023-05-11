import os
import librosa
import librosa.display
import librosa.beat
import sounddevice as sd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import spatial
from scipy.spatial import distance

# discartar a 1 e a ultima coluna dos top100

SR = 22050
MONO = True
warnings.filterwarnings("ignore")

def normalize(arrToNorm):
    maxs = np.max(arrToNorm, axis = 0)
    mins = np.min(arrToNorm, axis = 0)
    ans= np.array(arrToNorm)
    ans[np.arange(len(arrToNorm)),:] = (ans[np.arange(len(arrToNorm)),: ] - mins)/(maxs-mins)
    return ans

def statistics(feature):
    mean = np.mean(feature)
    desv = np.std(feature)
    skew = st.skew(feature)
    kurto = st.kurtosis(feature)
    median = np.median(feature)
    max_m = feature.max()
    min_m = feature.min() 
    
    return np.array([mean, desv, skew, kurto, median, max_m, min_m])

def features_librosa(dirname):    
    ans=np.array([])
    i=0
    for filename in os.listdir(dirname):
        print(filename, i)
        i+=1
        # spectral features
        y,fs = librosa.load(dirname+'/'+filename, sr = SR, mono = MONO)
        mfccs_file = librosa.feature.mfcc(y, sr=SR, n_mfcc=13)
        spcentroid = librosa.feature.spectral_centroid(y, sr=SR)
        spband = librosa.feature.spectral_bandwidth(y, sr=SR)
        spcontrast = librosa.feature.spectral_contrast(y, sr=SR)
        spflatness = librosa.feature.spectral_flatness(y)
        sprolloff = librosa.feature.spectral_rolloff(y, sr=SR)
        rms = librosa.feature.rms(y)
        zcr = librosa.feature.zero_crossing_rate(y)
        f0 = librosa.yin(y, sr=SR, fmin=20, fmax=11025)
        f0[f0==11025]=0
        all_features_array = np.vstack((mfccs_file, spcentroid, spband, spcontrast, spflatness, sprolloff, f0, rms, zcr))
        all_stats = np.apply_along_axis(statistics, 1, all_features_array).flatten()


        tempo = librosa.beat.tempo(y,sr=SR)
        aid = np.append(all_stats, tempo)
        if i==1:
            ans = np.array(aid)
        else:
            ans= np.vstack((ans,aid))

    ans = np.array(ans)
    return normalize(ans)

def euclidean_distance(a, b):
    return distance.euclidean(a, b)

def manhattan_distance(a, b):
    return distance.cityblock(a, b)

def cosine_distance(a, b):
    return distance.cosine(a, b)

def calculate_similarity_matrix(features, similarity_function):
    n = len(features)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            similarity_matrix[i, j] = similarity_function(features[i], features[j])
            similarity_matrix[j, i] = similarity_matrix[i, j] 

    return similarity_matrix

def create_rankings(similarity_matrix, top_k=20):
    rankings = []
    for i in range(similarity_matrix.shape[0]):
        ranked_indices = np.argsort(similarity_matrix[i])[:top_k]
        rankings.append(ranked_indices)
    return np.array(rankings)

def clean_features(features):
    features_cleaned = np.nan_to_num(features)
    features_cleaned = np.where(np.isinf(features_cleaned), 0, features_cleaned)
    return features_cleaned



def main():
    # 2.1.1. Ler o ficheiro e criar um array numpy com as features disponibilizadas.
    file_path = 'Features/top100_features.csv'

    if os.path.isfile(file_path):
        features_array = np.genfromtxt(file_path, delimiter=',', skip_header=1)[:,1:-1]

        # 2.1.2. Normalizar as features.
        normalized_features_array = normalize(features_array)

        # 2.1.3. Criar e gravar em ficheiro o array com as features extraídas linhas = músicas | colunas = valores das features.
        np.savetxt("Features/top100_features_normalized.csv", normalized_features_array, fmt = "%lf", delimiter= ",")

    else:
        print(f'O ficheiro {file_path} não foi encontrado.')
        
    audio_folder = 'Dataset/Musics'

    features_norm_obtained  = features_librosa(audio_folder)
    print(features_norm_obtained.shape)

    # 2.2.4. Criar e gravar em ficheiro o array numpy com as features extraídas.
    
    np.savetxt("Features/features_900_normalized.csv", features_norm_obtained, fmt = "%lf", delimiter= ",")

    top100_featuresN = np.loadtxt("Features/top100_features_normalized.csv", delimiter=",")
    features_900_normalized = np.loadtxt("Features/features_900_normalized.csv", delimiter=",")

    feature_sets = [
        top100_featuresN,
        features_900_normalized
    ]

    file_name_prefixes = [
        "top100",
        "features"
    ]

    for query_index, features in enumerate(feature_sets):
        # Clean the features array
        features_cleaned = clean_features(features)

        # Euclidean distance
        sim_matrix_euclidean = calculate_similarity_matrix(features_cleaned, euclidean_distance)
        np.savetxt(f"Features/{file_name_prefixes[query_index]}_euclidean.csv", sim_matrix_euclidean, fmt="%lf", delimiter=",")
        rankings_euclidean = create_rankings(sim_matrix_euclidean)
        np.savetxt(f"Features/{file_name_prefixes[query_index]}_rankings_euclidean.csv", rankings_euclidean, fmt='%d', delimiter=",")

        # Manhattan distance
        sim_matrix_manhattan = calculate_similarity_matrix(features_cleaned, manhattan_distance)
        np.savetxt(f"Features/{file_name_prefixes[query_index]}_manhattan.csv", sim_matrix_manhattan, fmt="%lf", delimiter=",")
        rankings_manhattan = create_rankings(sim_matrix_manhattan)
        np.savetxt(f"Features/{file_name_prefixes[query_index]}_rankings_manhattan.csv", rankings_manhattan, fmt='%d', delimiter=",")

        # Cosine distance
        sim_matrix_cosine = calculate_similarity_matrix(features_cleaned, cosine_distance)
        np.savetxt(f"Features/{file_name_prefixes[query_index]}_cosine.csv", sim_matrix_cosine, fmt="%lf", delimiter=",")
        rankings_cosine = create_rankings(sim_matrix_cosine)
        np.savetxt(f"Features/{file_name_prefixes[query_index]}_rankings_cosine.csv", rankings_cosine, fmt='%d', delimiter=",")


if __name__ == '__main__':
    main()
