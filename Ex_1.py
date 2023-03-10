import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy.fftpack import dct, idct
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
    y = y.astype('float32')
    if subsampling_ratio == "4:2:0":
        cb = cv2.resize(cb.astype('float32'), (cb.shape[1]//2, cb.shape[0]//2), interpolation=cv2.INTER_AREA)
        cr = cv2.resize(cr.astype('float32'), (cr.shape[1]//2, cr.shape[0]//2), interpolation=cv2.INTER_AREA)
    elif subsampling_ratio == "4:2:2":
        cb = cv2.resize(cb.astype('float32'), (cb.shape[1]//2, cb.shape[0]), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(cr.astype('float32'), (cr.shape[1]//2, cr.shape[0]), interpolation=cv2.INTER_LINEAR)
    
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
    #y = cv2.resize(y, (y.shape[1]*2, y.shape[0]*2), interpolation=cv2.INTER_LINEAR)
    
    return y, cb, cr


def calculate_dct(X):
    # Apply DCT to rows and columns of channel
    X_dct = dct(dct(X, norm="ortho").T, norm="ortho").T
    return X_dct

def calculate_idct(channel_dct):
    # Apply IDCT to rows and columns of channel
    channel_idct = idct(idct(channel_dct, norm='ortho').T, norm='ortho').T
    return channel_idct

def calculate_dct_blocks(matrix, size):
    #if matrix.shape[0] % size != 0 or matrix.shape[1] % size != 0:
    #    raise ValueError("Matrix shape must be a multiple of block size. Matrix size is " + str(matrix.shape) + " and block size is " + str(size) + ".")
    
    # Divide matrix into 8x8 blocks and apply DCT to each block
    dct_blocks = np.zeros(matrix.shape)
    for i in range(0, matrix.shape[0], size):
        for j in range(0, matrix.shape[1], size):
            block = matrix[i:i+size, j:j+size]
            dct_block = calculate_dct(block)
            dct_blocks[i:i+size, j:j+size] = dct_block
    

    return dct_blocks

def calculate_idct_blocks(dct_blocks, size):
    if dct_blocks.shape[0] % size != 0 or dct_blocks.shape[1] % size != 0:
        raise ValueError("Matrix shape must be a multiple of block size")

    idct_blocks = np.zeros(dct_blocks.shape)
    for i in range(0, dct_blocks.shape[0], size):
        for j in range(0, dct_blocks.shape[1], size):
            block = dct_blocks[i:i+size, j:j+size]
            idct_block = calculate_idct(block)
            idct_blocks[i:i+size, j:j+size] = idct_block

    return idct_blocks


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
    y = np.round(y).astype(int)
    cb = np.round(cb).astype(int)
    cr = np.round(cr).astype(int)

    #ycbcr para imagem usando matplotlib
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(y, cmap='gray')
    ax1.set_title('Componente Y')
    ax2.imshow(cb, cmap='gray')
    ax2.set_title('Componente Cb')
    ax3.imshow(cr, cmap='gray')
    ax3.set_title('Componente Cr')
    plt.show()

    y_d, cb_d, cr_d = subsample_yuv(y.astype(np.uint8), cb.astype(np.uint8), cr.astype(np.uint8), "4:2:0")

    #plot new image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(y_d, cmap='gray')
    ax1.set_title('Y downsampled 4:2:0')
    ax2.imshow(cb_d, cmap='gray')
    ax2.set_title('Cb downsampled 4:2:0')
    ax3.imshow(cr_d, cmap='gray')
    ax3.set_title('Cr downsampled 4:2:0')
    plt.show()

    # y_up, cb_up, cr_up = upsample_yuv(y_d, cb_d, cr_d, "4:2:0")

    # if np.allclose(y, y_up):
    #     print("Y is the same")
    # else:
    #     print("Y is different")

    y_dct = calculate_dct(y_d)
    cb_dct = calculate_dct(cb_d)
    cr_dct = calculate_dct(cr_d)

    Y_dct_log = np.log(np.abs(y_dct) + 0.0001)
    Cb_dct_log = np.log(np.abs(cb_dct) + 0.0001)
    Cr_dct_log = np.log(np.abs(cr_dct) + 0.0001)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(Y_dct_log, cmap='gray')
    ax1.set_title('Y DCT')
    ax2.imshow(Cb_dct_log, cmap='gray')
    ax2.set_title('Cb DCT')
    ax3.imshow(Cr_dct_log, cmap='gray')
    ax3.set_title('Cr DCT')
    plt.show()

    y_dct8 = calculate_dct_blocks(y_d, 8)
    cb_dct8 = calculate_dct_blocks(cb_d, 8)
    cr_dct8 = calculate_dct_blocks(cr_d, 8)

    Y_dct8_log = np.log(np.abs(y_dct8) + 0.0001)
    Cb_dct8_log = np.log(np.abs(cb_dct8) + 0.0001)
    Cr_dct8_log = np.log(np.abs(cr_dct8) + 0.0001)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(Y_dct8_log, cmap='gray')
    ax1.set_title('Y DCT 8x8')
    ax2.imshow(Cb_dct8_log, cmap='gray')
    ax2.set_title('Cb DCT 8x8')
    ax3.imshow(Cr_dct8_log, cmap='gray')
    ax3.set_title('Cr DCT 8x8')
    plt.show()
    






    return [y_dct8, cb_dct8, cr_dct8], [image.shape[0], image.shape[1]]

def decode(image, original_shape):

    y_d = calculate_idct_blocks(image[0], 8)
    cb_d = calculate_idct_blocks(image[1], 8)
    cr_d = calculate_idct_blocks(image[2], 8)

    Y, Cb, Cr = upsample_yuv(y_d, cb_d, cr_d, "4:2:0")


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

    padded_image = np.dstack((r, g, b))
    

    height, width = original_shape
    padded_height, padded_width = padded_image.shape[:2]
    w_pad = (padded_width - width)
    h_pad = (padded_height - height)
    image = padded_image[:-h_pad, :-w_pad]

    plt.imshow(image)
    plt.show()
    
    
    return


colormap = [[200, 0, 0], [0, 200, 0], [0, 0, 200]]
encoded_image, original_shape = encoder("barn_mountains.bmp", colormap)

decode(encoded_image, original_shape)
