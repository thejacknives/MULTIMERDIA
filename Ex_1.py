import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2

def pad_image_to_multiple_of_32(image):
    height, width, channels = image.shape
    new_height = (height + 31) // 32 * 32
    new_width = (width + 31) // 32 * 32
    padded_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    padded_image[:height, :width, :] = image

    last_column = padded_image[:, width-1, :].reshape(new_height, 1, channels)
    padded_image[:, width:, :] = np.repeat(last_column, new_width-width, axis=1)
    last_row = padded_image[height-1, :, :].reshape(1, new_width, channels)
    padded_image[height:, :, :] = np.repeat(last_row, new_height-height, axis=0)
    return padded_image

def unpad_image(image, original_height, original_width):
    return image[:original_height, :original_width, :]

def subsample_yuv(y, cb, cr, subsampling_ratio):
    
    # Subsample Cb and Cr channels
    if subsampling_ratio == "4:2:0":
        cb = cv2.resize(cb, (cb.shape[1]//2, cb.shape[0]//2), interpolation=cv2.INTER_AREA)
        cr = cv2.resize(cr, (cr.shape[1]//2, cr.shape[0]//2), interpolation=cv2.INTER_AREA)
    elif subsampling_ratio == "4:2:2":
        cb = cv2.resize(cb, (cb.shape[1]//2, cb.shape[0]), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(cr, (cr.shape[1]//2, cr.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return y, cb, cr

def upsample_yuv(y, cb, cr, upsampling_ratio):
    
    # Upsample Cb and Cr channels
    if upsampling_ratio == "4:2:0":
        cb = cv2.resize(cb, (cb.shape[1]*2, cb.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        cr = cv2.resize(cr, (cr.shape[1]*2, cr.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    elif upsampling_ratio == "4:2:2":
        cb = cv2.resize(cb, (cb.shape[1]*2, cb.shape[0]), interpolation=cv2.INTER_CUBIC)
        cr = cv2.resize(cr, (cr.shape[1]*2, cr.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Upsample Y channel using bilinear interpolation
    y = cv2.resize(y, (y.shape[1]*2, y.shape[0]*2), interpolation=cv2.INTER_LINEAR)
    
    return y, cb, cr


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
    # Load an example image
    image = plt.imread(image_path)

    # Pad the image
    padded_image = pad_image_to_multiple_of_32(image)

    # Display the original and padded images
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].set_title('Original image')
    axs[1].imshow(padded_image)
    axs[1].set_title('Padded image')
    plt.show()

    
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
    y = np.round(y).astype(np.uint8)
    cb = np.round(cb).astype(np.uint8)
    cr = np.round(cr).astype(np.uint8)

    #ycbcr para imagem usando matplotlib
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(y, cmap='gray')
    ax1.set_title('Componente Y')
    ax2.imshow(cb, cmap='gray')
    ax2.set_title('Componente Cb')
    ax3.imshow(cr, cmap='gray')
    ax3.set_title('Componente Cr')
    plt.show()

    y_d, cb_d, cr_d = subsample_yuv(y, cb, cr, "4:2:0")

    #plot new image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(y_d, cmap='gray')
    ax1.set_title('Y downsampled 4:2:0')
    ax2.imshow(cb_d, cmap='gray')
    ax2.set_title('Cb downsampled 4:2:0')
    ax3.imshow(cr_d, cmap='gray')
    ax3.set_title('Cr downsampled 4:2:0')
    plt.show()





    



    return [y, cb, cr]

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
