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
    colormap = colormap.astype(float) / 255.0
    
    indexed_image = np.zeros_like(rgb_image[:,:,0], dtype=int)
    for i, color in enumerate(colormap):
        dist = np.linalg.norm(rgb_image - color, axis=2)
        indexed_image[dist < np.linalg.norm(rgb_image - colormap[indexed_image], axis=2)] = i
    
    # Visualize image with colormap
    plt.imshow(indexed_image, cmap=plt.cm.colors.ListedColormap(colormap))
    plt.title("Encoded Image with Custom Colormap")
    plt.show()
    
    # Split RGB components
    r, g, b = np.split(rgb_image, 3, axis=2)
    
    # Visualize RGB components
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    plt.title("Original Image")
    
    plt.subplot(2, 2, 2)
    plt.imshow(r.squeeze(), cmap="Reds")
    plt.title("Red Component")
    
    plt.subplot(2, 2, 3)
    plt.imshow(g.squeeze(), cmap="Greens")
    plt.title("Green Component")
    
    plt.subplot(2, 2, 4)
    plt.imshow(b.squeeze(), cmap="Blues")
    plt.title("Blue Component")
    
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
colormap = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
encoded_image = encoder("peppers.bmp", colormap)
decoded_image = decode(encoded_image)

