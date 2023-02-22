from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_ycbcr(rgb_image):
    # Converte a imagem RGB para um array numpy
    rgb_array = np.array(rgb_image)

    # Cria um array vazio para a imagem YCbCr
    ycbcr_array = np.empty_like(rgb_array)

    # Converte cada pixel da imagem RGB para YCbCr
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            # Valores RGB do pixel
            r = rgb_array[i, j, 0]
            g = rgb_array[i, j, 1]
            b = rgb_array[i, j, 2]

            # Calcula os valores YCbCr
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

            # Guarda os valores YCbCr no array
            ycbcr_array[i, j, 0] = y
            ycbcr_array[i, j, 1] = cb
            ycbcr_array[i, j, 2] = cr

    # Converte o array para uma imagem e retorna
    ycbcr_image = Image.fromarray(ycbcr_array, mode='YCbCr')
    return ycbcr_image

def ycbcr_to_rgb(ycbcr_image):
    ycbcr_array = np.array(ycbcr_image)

    rgb_array = np.empty_like(ycbcr_array)

    for i in range(ycbcr_array.shape[0]):
        for j in range(ycbcr_array.shape[1]):
            y = ycbcr_array[i, j, 0]
            cb = ycbcr_array[i, j, 1]
            cr = ycbcr_array[i, j, 2]

            r = y + 1.402 * (cr - 128)
            g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
            b = y + 1.772 * (cb - 128)

            rgb_array[i, j, 0] = r
            rgb_array[i, j, 1] = g
            rgb_array[i, j, 2] = b

    rgb_image = Image.fromarray(np.uint8(rgb_array), mode='RGB')
    return rgb_image

rgb_image = Image.open("peppers.jpg")
ycbcr_image = rgb_to_ycbcr(rgb_image)
ycbcr_image.show()
rgb_image = ycbcr_to_rgb(ycbcr_image)
rgb_image.show()

#Inversão imperfeita, devido a perda de informação na conversão