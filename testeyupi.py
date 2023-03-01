import io
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def encoder(image_path, colormap):
    # Load image
    image = Image.open(image_path)
    rgb_image = np.array(image)
    
    # Implement colormap
    colormap = np.array(colormap)
    colormap = colormap.astype(float)
    
    indexed_image = np.zeros_like(rgb_image[:,:,0], dtype=int)
    for i, color in enumerate(colormap):
        dist = np.linalg.norm(rgb_image - color, axis=2)
        indexed_image[dist < np.linalg.norm(rgb_image - colormap[indexed_image], axis=2)] = i
    
    # Visualize image with colormap
    plt.imshow(indexed_image, cmap=plt.cm.colors.ListedColormap(colormap))
    plt.title("Encoded Image with Custom Colormap")
    plt.show()
    
    # Split image into RGB components using numpy indexing
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]

    # Create color images for each channel with zeros in the other channels
    red_image = np.zeros_like(rgb_image)
    red_image[:, :, 0] = r

    green_image = np.zeros_like(rgb_image)
    green_image[:, :, 1] = g

    blue_image = np.zeros_like(rgb_image)
    blue_image[:, :, 2] = b

    # Visualize RGB components
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


    # salva a imagem compactada como um objeto BytesIO
    # salva a imagem compactada como um objeto BytesIO
    compressed_image = io.BytesIO()
    image.save(compressed_image, "BMP")

    return compressed_image

def decode(encoded_image_path):
    # Abre a imagem codificada
    encoded_image = Image.open(encoded_image_path)

    # Obtém a largura e altura da imagem
    width, height = encoded_image.size

    # Cria um objeto Image vazio para a imagem decodificada
    decoded_image = Image.new("RGB", (width, height))

    # Itera através de cada pixel e decodifica a cor original a partir do canal verde do pixel codificado
    for x in range(width):
        for y in range(height):
            encoded_pixel = encoded_image.getpixel((x, y))
            decoded_pixel = (encoded_pixel[1], encoded_pixel[1], encoded_pixel[1])
            decoded_image.putpixel((x, y), decoded_pixel)

    # Visualiza a imagem decodificada
    plt.imshow(np.array(decoded_image))
    plt.title("Decoded Image")
    plt.show()

    return decoded_image


#run functions with peppers.bmp
colormap = np.array( [[1, 0, 0], [0, 1, 0], [0, 0, 1]] ) 
encoded_image = encoder("barn_mountains.bmp", colormap)
decoded_image = decode(encoded_image)