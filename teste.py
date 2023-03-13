import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy.fftpack import dct, idct
import cv2

quality_factor = 50

quantization_matrix_Y = np.array(
    [[16, 11, 10, 16, 24, 40, 41, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

quantization_matrix_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                         [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])

sub_sampling = "4:2:0"


matriz_conversao = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
    ])

def clamping(matriz, maximo, minimo):   
    matriz[matriz>maximo] = maximo
    matriz[matriz<minimo] = minimo
    return matriz.astype('uint8')

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

        # Display the original and padded images
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].set_title('Original image')
    axs[1].imshow(padded_image)
    axs[1].set_title('Padded image')
    plt.show()

    return padded_image

def unpad_image(image, original_height, original_width):
    return image[:original_height, :original_width, :]

def subsample_yuv(y, cb, cr, subsampling_ratio):
    # Subsample Cb and Cr channels
    y_d = y.astype('float32')
    if subsampling_ratio == "4:2:0":
        cb_d = cv2.resize(cb.astype('float32'), (cb.shape[1]//2, cb.shape[0]//2), interpolation=cv2.INTER_AREA)
        cr_d = cv2.resize(cr.astype('float32'), (cr.shape[1]//2, cr.shape[0]//2), interpolation=cv2.INTER_AREA)
    elif subsampling_ratio == "4:2:2":
        cb_d = cv2.resize(cb.astype('float32'), (cb.shape[1]//2, cb.shape[0]), interpolation=cv2.INTER_LINEAR)
        cr_d = cv2.resize(cr.astype('float32'), (cr.shape[1]//2, cr.shape[0]), interpolation=cv2.INTER_LINEAR)

    #plot new image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(y_d, cmap='gray')
    ax1.set_title('Y downsampled 4:2:0')
    ax2.imshow(cb_d, cmap='gray')
    ax2.set_title('Cb downsampled 4:2:0')
    ax3.imshow(cr_d, cmap='gray')
    ax3.set_title('Cr downsampled 4:2:0')
    plt.show()
    
    return y_d, cb_d, cr_d

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

def calculate_single_channel_dct(channel):
    channel_dct = dct(dct(channel, norm="ortho").T, norm="ortho").T
    return channel_dct

def calculate_dct(y, cb, cr):
    # Apply DCT to rows and columns of channel
    y_dct = dct(dct(y, norm="ortho").T, norm="ortho").T
    cb_dct = dct(dct(cb, norm="ortho").T, norm="ortho").T
    cr_dct = dct(dct(cr, norm="ortho").T, norm="ortho").T

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
    

    return y_dct, cb_dct, cr_dct

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
            dct_block = calculate_single_channel_dct(block)
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

def divide_rgb(image):
    # DIVIDE NOS COMPONENTES RGB
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    cmred =clr.LinearSegmentedColormap.from_list('red',[(0,0,0), (1,0,0)],256)
    cmgreen = clr.LinearSegmentedColormap.from_list('green',[(0,0,0), (0,1,0)],256)
    cmblue = clr.LinearSegmentedColormap.from_list('blue',[(0,0,0), (0,0,1)],256)
    cmgray = clr.LinearSegmentedColormap.from_list('grey',[(0,0,0), (1,1,1)],256)

    # VE COM OS CHANNELS CRIADOS EM CIMA
    plt.subplot(2, 2, 1)
    plt.imshow(image)
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

    return r, g, b

def rgb_to_ycbcr(r,g,b):
    #rgb para ycbcr
    #matriz de conversao
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

    return y, cb, cr

def quantizacao_dpcm(canal_dct, quantization_matrix, quality_factor):
    if quality_factor >= 50:
        scale_factor = ((100 - quality_factor) / 50)
    else:
        scale_factor = 50 / quality_factor
    if scale_factor != 0:
        true_quantization_matrix = np.ceil(scale_factor * quantization_matrix)
        true_quantization_matrix = clamping(true_quantization_matrix, 255, 1)
    else:
        true_quantization_matrix = quantization_matrix
    size = canal_dct.shape
    output = np.zeros(canal_dct.shape)
    valor_anterior = 0
    aux_anterior = 0
    for lin in range(0, size[0], 8):
        for col in range(0, size[1], 8):
            block = canal_dct[lin:lin + 8, col:col + 8]
            if quality_factor != 100:
                quantized_block = np.divide(block, true_quantization_matrix)
                quantized_block = np.round(quantized_block).astype(int)
            else:
                quantized_block = np.round(block).astype(int)
            # EXERCICIO 9 #
            aux_anterior = quantized_block[0][0]
            quantized_block[0][0] = quantized_block[0][0] - valor_anterior
            valor_anterior = aux_anterior
            # # # # # # # #
            output[lin:lin + 8, col:col + 8] = quantized_block  
    output = np.round(output).astype(int)
    
    return output

def inv_quantizacao_dpcm(canal_dct, quantization_matrix, quality_factor):
    if quality_factor >= 50:
        scale_factor = ((100 - quality_factor) / 50)
    else:
        scale_factor = 50 / quality_factor
    if scale_factor != 0:
        true_quantization_matrix = np.ceil(scale_factor * quantization_matrix)
        true_quantization_matrix = clamping(true_quantization_matrix, 255, 1)
    else:
        true_quantization_matrix = quantization_matrix
    size = canal_dct.shape
    output = np.zeros(canal_dct.shape)
    valor_anterior = 0
    for lin in range(0, size[0], 8):
        for col in range(0, size[1], 8):
            block = canal_dct[lin:lin + 8, col:col + 8]
            # EXERCICIO 9 #
            block[0][0] = block[0][0] + valor_anterior
            valor_anterior = block[0][0]
            # # # # # # # #
            if quality_factor != 100:
                quantized_block = np.multiply(block, true_quantization_matrix)
            else:
                quantized_block = block
            output[lin:lin + 8, col:col + 8] = quantized_block
    output = output.astype(float)

    return output


def encoder(image_path, size=(32, 32)):
   
    #padding
    # Load an example image
    image = plt.imread(image_path)
    print(image.dtype)

    # Pad the image
    padded_image = pad_image_to_multiple_of_32(image)

    r, g, b = divide_rgb(padded_image)

    y, cb, cr = rgb_to_ycbcr(r, g, b)

    y_d, cb_d, cr_d = subsample_yuv(y.astype(np.uint8), cb.astype(np.uint8), cr.astype(np.uint8), sub_sampling)

    y_dct, cb_dct, cr_dct = calculate_dct(y_d, cb_d, cr_d)

    block_size = 8

    y_dct8 = calculate_dct_blocks(y_d, block_size)
    cb_dct8 = calculate_dct_blocks(cb_d, block_size)
    cr_dct8 = calculate_dct_blocks(cr_d, block_size)

    y_dct8_quant = quantizacao_dpcm(y_dct8, quantization_matrix_Y, quality_factor)
    cb_dct8_quant = quantizacao_dpcm(cb_dct8, quantization_matrix_CbCr, quality_factor)
    cr_dct8_quant = quantizacao_dpcm(cr_dct8, quantization_matrix_CbCr, quality_factor)

   
    
    print("y_dct8:",y_dct8[8:16,8:16])

    #show the 3 images in same plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(np.log10(np.absolute(y_dct8_quant) + 0.0001), cmap='gray')
    ax1.set_title('Y DCT 8x8 Quantizado')
    ax2.imshow(np.log10(np.absolute(cb_dct8_quant) + 0.0001), cmap='gray')
    ax2.set_title('Cb DCT 8x8 Quantizado')
    ax3.imshow(np.log10(np.absolute(cr_dct8_quant) + 0.0001), cmap='gray')
    ax3.set_title('Cr DCT 8x8 Quantizado')
    plt.show()
    

    return [y_dct8_quant, cb_dct8_quant, cr_dct8_quant], [image.shape[0], image.shape[1]],

def decode(image, original_shape):

    y_d = inv_quantizacao_dpcm(image[0], quantization_matrix_Y, quality_factor)
    cb_d = inv_quantizacao_dpcm(image[1], quantization_matrix_CbCr, quality_factor)
    cr_d = inv_quantizacao_dpcm(image[2], quantization_matrix_CbCr, quality_factor)
    block_size = 8

    y_d = calculate_idct_blocks(image[0], block_size)
    cb_d = calculate_idct_blocks(image[1], block_size)
    cr_d = calculate_idct_blocks(image[2], block_size)

    Y, Cb, Cr = upsample_yuv(y_d, cb_d, cr_d, sub_sampling)


    # matriz_conversao = np.array([
    #     [1.0, 0.0, 1.402],
    #     [1.0, -0.344136, -0.714136],
    #     [1.0, 1.772, 0.0]
    #     ])
    imc = np.linalg.inv(matriz_conversao)

    r = Y * imc[0][0] + imc[0][1] * (Cb - 128) + imc[0][2] * (Cr - 128)
    g = Y * imc[1][0] + imc[1][1] * (Cb - 128) + imc[1][2] * (Cr - 128)
    b = Y * imc[2][0] + imc[2][1] * (Cb - 128) + imc[2][2] * (Cr - 128)

    r = np.clip(r, 0, 255).round().astype(np.uint8)
    g = np.clip(g, 0, 255).round().astype(np.uint8)
    b = np.clip(b, 0, 255).round().astype(np.uint8)

    padded_image = np.dstack((r, g, b))
    

    height, width = original_shape
    padded_height, padded_width = padded_image.shape[:2]
    w_pad = (padded_width - width)
    h_pad = (padded_height - height)
    image = padded_image[:-h_pad, :-w_pad]

    plt.imshow(image)
    plt.show()
    
    
    return

encoded_image, original_shape = encoder("barn_mountains.bmp")

decode(encoded_image, original_shape)
