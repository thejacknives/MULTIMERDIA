from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_ycbcr(rgb_image):

    # DIVIDE NOS COMPONENTES RGB
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]

    matriz_conversao = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
    ])

    y = r * matriz_conversao[0][0] + g * matriz_conversao[0][1] + b * matriz_conversao[0][2]
    cb = matriz_conversao[1][0] * r + (matriz_conversao[1][1]) * g + matriz_conversao[1][2] * b + 128
    cr = matriz_conversao[2][0] * r + (matriz_conversao[2][1] * g) + (matriz_conversao[2][2] * b) + 128

    #arredondamentos
    y = np.round(y).astype(int)
    cb = np.round(cb).astype(int)
    cr = np.round(cr).astype(int)

    #guarda os valores num array
    ycbcr_array = np.stack([y, cb, cr], axis=-1)

    # Converte o array para uma imagem e DA RETURN
    ycbcr_image = Image.fromarray(ycbcr_array, mode='YCbCr')

    y, cb, cr = ycbcr_image.split()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(y, cmap='gray')
    ax1.set_title('Componente Y')
    ax2.imshow(cb, cmap='gray')
    ax2.set_title('Componente Cb')
    ax3.imshow(cr, cmap='gray')
    ax3.set_title('Componente Cr')
    plt.show()

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

image = cv2.imread("barn_mountains.bmp")
rgb_image = np.array(image)
rgb_to_ycbcr(rgb_image)
