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

sr = 22050
mono = True
warnings.filterwarnings("ignore")

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

def extract_features(audio_file):
    y, fs = librosa.load(audio_file, sr=sr, mono=mono)

    # Features Espectrais
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y)[0, :]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)[0, :]
    spectral_contrast = librosa.feature.spectral_contrast(y=y)
    print(spectral_contrast.shape)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0, :]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y)[0, :]

    # Features Temporais
    f0 = librosa.yin(y=y, fmin=20, fmax=fs / 2)
    f0[f0 == fs / 2] = 0
    rms = librosa.feature.rms(y=y)[0, :]
    zero_cross = librosa.feature.zero_crossing_rate(y=y)[0, :]

    # Outras features
    time = librosa.beat.tempo(y=y)

    features = [
        mfcc,
        spectral_centroid,
        spectral_bandwidth,
        spectral_contrast,
        spectral_flatness,
        spectral_rolloff,
        f0,
        rms,
        zero_cross,
        time[0]
    ]

    stats = []

    for feature in features:
        if len(feature.shape) > 1:
            for i in range(feature.shape[0]):
                stats.extend(statistics(feature[i, :]))
        else:
            stats.extend(statistics(feature))

    return np.array(stats)

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
        for j in range(n):
            similarity_matrix[i, j] = similarity_function(features[i], features[j])

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
        features_array = np.genfromtxt(file_path, delimiter=',', skip_header=1)

        # 2.1.2. Normalizar as features no intervalo [0, 1].
        normalized_features_array = normalize(features_array)

        # 2.1.3. Criar e gravar em ficheiro um array numpy com as features extraídas (linhas =
        # músicas; colunas = valores das features).
        np.savetxt("Features/top100_features_normalized.csv", normalized_features_array, fmt = "%lf", delimiter= ",")

    else:
        print(f'O ficheiro {file_path} não foi encontrado.')
        
    audio_folder = 'Queries'
    audio_files = [file for file in os.listdir(audio_folder) if file.endswith('.mp3')]

    stat = np.zeros((900, 196), dtype=np.float64)

    for index, audio_file in enumerate(audio_files):
        file_path = os.path.join(audio_folder, audio_file)
        features = extract_features(file_path)
        
        stat[index] = features

    # 2.2.3. Normalizar as features no intervalo [0, 1].
    normalized_features_array = normalize(stat)
    # 2.2.4. Criar e gravar em ficheiro o array numpy com as features extraídas.
    
    np.savetxt("Features/features_900_normalized.csv", normalized_features_array, fmt = "%lf", delimiter= ",")

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
