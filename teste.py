import numpy as np
from PIL import Image

def encoder(file_path, colormap):
    # Carrega a imagem BMP
    img = Image.open(file_path)
    
    # Converte a imagem para um array numpy
    img_arr = np.array(img)
    
    # Aplica o colormap
    img_cmap = colormap[img_arr]
    
    # Mostra a imagem com o colormap aplicado
    Image.fromarray(img_cmap).show()
    
    # Separa a imagem nos seus componentes RGB
    r = img_cmap[:, :, 0]
    g = img_cmap[:, :, 1]
    b = img_cmap[:, :, 2]
    
    # Retorna os componentes RGB separados
    return r, g, b

# Exemplo de uso
colormap = np.array([[0, 0, 0], [1, 1, 1]])
r, g, b = encoder("peppers.bmp", colormap)
