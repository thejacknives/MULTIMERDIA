import numpy as np
import cv2
import matplotlib.pyplot as plt

def pad_image(image, size=(32, 32)):
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
    return padded_image


def unpad_image(image, original_shape):
    height, width = original_shape[:2]
    return image[:height, :width]

# Carrega a imagem
image = cv2.imread('sitio.bmp')

# Faz o padding
padded_image = pad_image(image)

# Mostra a imagem com padding
plt.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
plt.show()
