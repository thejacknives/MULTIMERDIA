import io
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def encoder(image_path, colormap, size=(32, 32)):
    image = Image.open(image_path)
    rgb_image = np.array(image)
    colormap = np.array(colormap)
    colormap = colormap.astype(float) / 255.0
    indexed_image = np.zeros_like(rgb_image[:,:,0], dtype=int)
    #ESTA A APLICAR O COLORMAP NA IMAGEM ORIGINAL
    for i, color in enumerate(colormap):
        dist = np.linalg.norm(rgb_image - color, axis=2)
        indexed_image[dist < np.linalg.norm(rgb_image - colormap[indexed_image], axis=2)] = i
    
    plt.imshow(indexed_image, cmap=plt.cm.colors.ListedColormap(colormap))
    plt.title("Imagem com o nosso belo Colormap")
    plt.show()

    # DIVIDE NOS COMPONENTES RGB
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]

    # SEPARA OS CHANNELS
    red_image = np.zeros_like(rgb_image)
    red_image[:, :, 0] = r

    green_image = np.zeros_like(rgb_image)
    green_image[:, :, 1] = g

    blue_image = np.zeros_like(rgb_image)
    blue_image[:, :, 2] = b

    # VE COM OS CHANNELS CRIADOS EM CIMA
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(red_image)
    plt.title("Red Component (Reds Colormap)")

    plt.subplot(2, 2, 3)
    plt.imshow(green_image)
    plt.title("Green Component (Greens Colormap)")

    plt.subplot(2, 2, 4)
    plt.imshow(blue_image)
    plt.title("Blue Component (Blues Colormap)")

    plt.show()
    # guarda a imagem compactada em BytesIO
    compressed_image = io.BytesIO()
    image.save(compressed_image, "BMP")

    #PADDING

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    new_height = int(np.ceil(height / size[0])) * size[0]
    new_width = int(np.ceil(width / size[1])) * size[1]
    if new_height < height:
        new_height += size[0]
    if new_width < width:
        new_width += size[1]
    pad_height = new_height - height
    pad_width = new_width - width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    padded_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    padded_image[top:-bottom, left:-right] = image
    padded_image[:top, left:-right] = image[0]
    padded_image[-bottom:, left:-right] = image[-1]
    padded_image[top:-bottom, :left] = image[:, 0:1]
    padded_image[top:-bottom, -right:] = image[:, -1:]
    plt.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
    plt.show()

    #rgb para ycbcr
    rgb_image = Image.open(image_path)
    # manda a imagem RGB para um array numpy
    rgb_array = np.array(rgb_image)

    ycbcr_array = np.empty_like(rgb_array)

    # converte cada pixel RGB para YCbCr
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            r = rgb_array[i, j, 0]
            g = rgb_array[i, j, 1]
            b = rgb_array[i, j, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
            ycbcr_array[i, j, 0] = y
            ycbcr_array[i, j, 1] = cb
            ycbcr_array[i, j, 2] = cr

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
    for x in range(width):
        for y in range(height):
            encoded_pixel = encoded_image.getpixel((x, y))
            decoded_pixel = (encoded_pixel[1], encoded_pixel[1], encoded_pixel[1])
            decoded_image.putpixel((x, y), decoded_pixel)

    plt.imshow(np.array(decoded_image))
    plt.title("Decoded Image")
    plt.show()

    height, width = original_shape[:2]
    padded_height, padded_width = padded_image.shape[:2]
    w_pad = (padded_width - width) // 2
    h_pad = (padded_height - height) // 2
    unpadded_image = padded_image[h_pad:-h_pad, w_pad:-w_pad]
    plt.imshow(cv2.cvtColor(unpadded_image, cv2.COLOR_BGR2RGB))
    plt.title("Unpadded Image")
    plt.show()

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





