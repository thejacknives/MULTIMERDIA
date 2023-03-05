import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from PIL import Image
import cv2

def encoder(image_path, colormap, size=(32, 32)):
    """
    image = Image.open(image_path)
    rgb_image = np.array(image)
    colormap = np.array(colormap)
    colormap = colormap.astype(float) / 255.0                                   ### NS SE E PRECISO MOSTRAR A IMAGEM ANTES COM O COLORMAP TODO FODIDO ###
    indexed_image = np.zeros_like(rgb_image[:,:,0], dtype=int)
    #ESTA A APLICAR O COLORMAP NA IMAGEM ORIGINAL
    for i, color in enumerate(colormap):
        dist = np.linalg.norm(rgb_image - color, axis=2)
        indexed_image[dist < np.linalg.norm(rgb_image - colormap[indexed_image], axis=2)] = i
    
    plt.imshow(indexed_image, cmap=plt.cm.colors.ListedColormap(colormap))
    plt.title("Imagem com o nosso belo Colormap")
    plt.show()
    """
    #padding
    image = plt.imread(image_path)
    width, height = image.shape
    padded_image = np.zeros((height + 2, width + 2, 3), dtype=np.uint8)

    padded_image = Image.fromarray(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
    # normaliza a imagem para poder ser np array para colormap :)) 
    padded_image = np.array(padded_image)
    
    # DIVIDE NOS COMPONENTES RGB
    r = padded_image[:, :, 0]
    g = padded_image[:, :, 1]
    b = padded_image[:, :, 2]

    cmred =clr.LinearSegmentedColormap.from_list('red',[(0,0,0), (1,0,0)],256)
    cmgreen = clr.LinearSegmentedColormap.from_list('green',[(0,0,0), (0,1,0)],256)
    cmblue = clr.LinearSegmentedColormap.from_list('blue',[(0,0,0), (0,0,1)],256)
    cmgray = clr.LinearSegmentedColormap.from_list('grey',[(0,0,0), (1,1,1)],256)

    # VE COM OS CHANNELS CRIADOS EM CIMA
    plt.subplot(2, 2, 1)
    plt.imshow(padded_image)
    plt.title("Original Image (RGB) (Imagem original)")

    plt.subplot(2, 2, 2)
    plt.imshow(r, cmred)
    plt.title("Red Component (Reds Colormap) (El componente rojo)")

    plt.subplot(2, 2, 3)
    plt.imshow(g, cmgreen)
    plt.title("Green Component (Greens Colormap) (El componente verde)")

    plt.subplot(2, 2, 4)
    plt.imshow(b, cmblue)
    plt.title("Blue Component (Blues Colormap) (El componente azul)")

    plt.show()
    # guarda a imagem compactada em BytesIO
    compressed_image = io.BytesIO()
    image.save(compressed_image, "BMP")



    #rgb para ycbcr
    #matriz de conversao
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
    ycbcr_array = np.stack([y, cb, cr], axis=-1).astype(np.uint8)

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




    return [compressed_image, padded_image, ycbcr_image]

def decode(encoded_image_path, padded_image, original_shape, ycbcr_image):
    encoded_image = Image.open(encoded_image_path)

    width, height = encoded_image.size

    decoded_image = Image.new("RGB", (width, height))

    # itera cada pixel e da decode a cor inicial
    """"""
    for x in range(width):
        for y in range(height):
            encoded_pixel = encoded_image.getpixel((x, y))
            decoded_pixel = (encoded_pixel[1], encoded_pixel[1], encoded_pixel[1])
            decoded_image.putpixel((x, y), decoded_pixel)
            height, width = original_shape[:2]

    padded_height, padded_width = padded_image.shape[:2]
    w_pad = (padded_width - width)
    h_pad = (padded_height - height)
    unpadded_image = padded_image[:-h_pad, :-w_pad]

    plt.imshow(cv2.cvtColor(unpadded_image, cv2.COLOR_BGR2RGB))
    plt.title("Unpadded Image")
    plt.show()

    ycbcr_image = np.array(ycbcr_image)
    Y = ycbcr_image[:, :, 0]
    Cb = ycbcr_image[:, :, 1]
    Cr = ycbcr_image[:, :, 2]

    # matriz_conversao = np.array([
    #     [1.0, 0.0, 1.402],
    #     [1.0, -0.344136, -0.714136],
    #     [1.0, 1.772, 0.0]
    #     ])
    matriz_conversao = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
    ])
    matriz_conversao = np.linalg.inv(matriz_conversao)

    r = Y * matriz_conversao[0][0] + matriz_conversao[0][1] * (Cb - 128) + matriz_conversao[0][2] * (Cr - 128)
    g = Y * matriz_conversao[1][0] + matriz_conversao[1][1] * (Cb - 128) + matriz_conversao[1][2] * (Cr - 128)
    b = Y * matriz_conversao[2][0] + matriz_conversao[2][1] * (Cb - 128) + matriz_conversao[2][2] * (Cr - 128)

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    r= np.round(r).astype(int)
    g = np.round(g).astype(int)
    b = np.round(b).astype(int)

    rgb_array = np.stack([r, g, b], axis=-1).astype(np.uint8) 
    rgb_image = Image.fromarray(np.uint8(rgb_array), mode='RGB')

    r, g, b = rgb_image.split()


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(r, cmap='gray')
    ax1.set_title('Componente R')
    ax2.imshow(g, cmap='gray')
    ax2.set_title('Componente G')
    ax3.imshow(b, cmap='gray')
    ax3.set_title('Componente B')
    plt.show()

    return [decoded_image, padded_image[:height, :width], rgb_image]


colormap = [[200, 0, 0], [0, 200, 0], [0, 0, 200]]
encoded_image, padded_image, ycbcr_image = encoder("barn_mountains.bmp", colormap)


original_shape = np.array(Image.open("barn_mountains.bmp")).shape
decoded_image, unpadded_image, rgb_image = decode(encoded_image, padded_image, original_shape, ycbcr_image)
