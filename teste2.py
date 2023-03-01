import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct, idct


matrix_Q_Y = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]])


matrix_Q_CbCr = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]])


def quantize(dct, matrix_Q):
    return np.round(dct / matrix_Q)


def dequantize(quantizedMatrix, matrix_Q):
    return np.round(quantizedMatrix * matrix_Q)


def padImage32(image):
    [nl, nc] = image.shape
    nel = 32-nl % 32
    nec = 32-nc % 32
    # get last line
    # extend matrix
    ll = image[nl-1, :][np.newaxis, :]
    # create repetition matrix
    repl = ll.repeat(nel, axis=0)
    # pad matrix
    imagePadded = np.vstack((image, repl))

    # get last column
    # extend matrix
    lc = imagePadded[:, nc-1][:, np.newaxis]
    # create repetition matrix
    repc = lc.repeat(nec, axis=1)
    # pad matrix
    imagePaddedBoth = np.hstack((imagePadded, repc))

    return imagePaddedBoth, [nl, nc]


def unpadImage(image, size):
    image = image[:size[0], :size[1]]
    return image


def readImage(image):
    return plt.imread(image)


def showImage(image, cmap=None):
    plt.imshow(image, cmap)


def rgbChannels(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return R, G, B


def rgbChannelsInverse(R, G, B):
    return np.dstack((R, G, B))


# plt.figure == abre uma nova janela
# img.shape()
ycbcr_matrix = np.array([
    [65.481, 128.553, 24.966],
    [-37.797, -74.203, 112.0],
    [112.0, -93.786, -18.214]
])


# Exercício 5
def rgb_to_ycbcr(rgb_image):
    # Define a matriz de transformação

    # Realiza a multiplicação matricial para obter a imagem YCbCr
    ycbcr = np.dot(rgb_image, ycbcr_matrix.T)

    G = ycbcr[:, :, 1] + 128
    B = ycbcr[:, :, 2] + 128

    return ycbcr


def ycbcr_to_rgb(ycbcr_image, show):
    # Define a matriz de transformação inversa
    inverse_transform = np.linalg.inv(ycbcr_matrix)

    G = ycbcr_image[:, :, 1] - 128
    B = ycbcr_image[:, :, 2] - 128

    # Realiza a multiplicação matricial para obter a imagem RGB
    rgb = np.dot(ycbcr_image, inverse_transform.T)

    # Normaliza a imagem RGB para o intervalo [0, 255]
    rgb[rgb < 0] = 0
    rgb[rgb > 255] = 255

    normalized_rgb = np.round(rgb).astype(np.uint8)
    if show:
        plt.figure()
        showImage(rgb)

    return normalized_rgb


def showCanals(ycbcr_image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plota o canal Y com colormap gray
    ax1.imshow(ycbcr_image[:, :, 0], cmap='gray')
    ax1.set_title("Canal Y")

    # Plota o canal Cb com colormap jet
    ax2.imshow(ycbcr_image[:, :, 1], cmap='gray')
    ax2.set_title("Canal Cb")

    # Plota o canal Cr com colormap jet
    ax3.imshow(ycbcr_image[:, :, 2], cmap='gray')
    ax3.set_title("Canal Cr")


def getYrefFcrFcb(string):
    return string.split(":")


def downSample(image, factor):
    valuesOfDownSample = getYrefFcrFcb(factor)
    Yref = valuesOfDownSample[0]
    Fcr = valuesOfDownSample[1]
    Fcb = valuesOfDownSample[2]

    if int(Fcb) == 0:
        downSampleImageCr = cv2.resize(
            image[:, :, 1], None, fx=1/int(Fcr), fy=1/int(Fcr))
        downSampleImageCb = cv2.resize(
            image[:, :, 2], None, fx=1/int(Fcr), fy=1/int(Fcr))
    else:
        downSampleImageCr = cv2.resize(
            image[:, :, 1], None, fx=1/int(Fcr), fy=1)
        downSampleImageCb = cv2.resize(
            image[:, :, 2], None, fx=1/int(Fcb), fy=1)

    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 3, 1)
    showImage(image[:, :, 0], "gray")
    fig.add_subplot(1, 3, 2)
    showImage(downSampleImageCr, "gray")
    fig.add_subplot(1, 3, 3)
    showImage(downSampleImageCb, "gray")

    return np.dstack((downSampleImageCb, downSampleImageCr)), factor


def upSample(Y_d, image, factor):
    valuesOfDownSample = getYrefFcrFcb(factor)
    Yref = valuesOfDownSample[0]
    Fcr = valuesOfDownSample[1]
    Fcb = valuesOfDownSample[2]

    if int(Fcb) == 0:
        upSampleImageCb = cv2.resize(
            image[:, :, 0], None, fx=int(Fcr), fy=int(Fcr))
        upSampleImageCr = cv2.resize(
            image[:, :, 1], None, fx=int(Fcr), fy=int(Fcr))

    else:
        upSampleImageCb = cv2.resize(image[:, :, 0], None, fx=int(Fcr), fy=1)
        upSampleImageCr = cv2.resize(image[:, :, 1], None, fx=int(Fcr), fy=1)

    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 3, 1)
    showImage(Y_d, "gray")
    fig.add_subplot(1, 3, 2)
    showImage(upSampleImageCr, "gray")
    fig.add_subplot(1, 3, 3)
    showImage(upSampleImageCb, "gray")

    return np.dstack((Y_d, upSampleImageCb, upSampleImageCr)), factor


def DCT(channel):
    return dct(dct(channel, norm='ortho').T, norm='ortho').T


def IDCT(channel):
    return idct(idct(channel, norm='ortho').T, norm='ortho').T


def logDCT(dct):
    return np.log(abs(dct) + 0.0001)

# tuple (r,g,b) de 0 a 1
# color "grey" "green" "red"


# do a function to do DTC in 8*8 blocks
def DCTBSxBS(channel, n):
    shape = channel.shape
    for i in range(int(shape[0]/n)):
        for j in range(int(shape[1]/n)):
            channel[i*n:(i+1)*n, j*n:(j+1) *
                    n] = DCT(channel[i*n:(i+1)*n, j*n:(j+1)*n])
    return channel


def IDCTBSxBS(channel, n):
    shape = channel.shape
    for i in range(int(shape[0]/n)):
        for j in range(int(shape[1]/n)):
            channel[i*n:(i+1)*n, j*n:(j+1) *
                    n] = IDCT(channel[i*n:(i+1)*n, j*n:(j+1)*n])
    return channel


def QBSxBS(channel, n, quantization_matrix):
    shape = channel.shape
    for i in range(int(shape[0]/n)):
        for j in range(int(shape[1]/n)):
            channel[i*n:(i+1)*n, j*n:(j+1) * n] = quantize((channel[i *
                                                                    n:(i+1)*n, j*n:(j+1)*n]), quantization_matrix)
    return channel


def IQBSxBS(channel, n, quantization_matrix):
    shape = channel.shape
    for i in range(int(shape[0]/n)):
        for j in range(int(shape[1]/n)):
            channel[i*n:(i+1)*n, j*n:(j+1) * n] = dequantize((channel[i *
                                                                      n:(i+1)*n, j*n:(j+1)*n]), quantization_matrix)
    return channel


def colorMap(color, tupleBegin, tupleEnd):
    return clr.LinearSegmentedColormap.from_list(color, [tupleBegin, tupleEnd], 256)

