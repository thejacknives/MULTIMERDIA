import numpy as np

def subsample(image, subsampling):
    # subsampling: tuple with 3 values indicating the subsampling ratio for Y, Cb and Cr
    height, width, channels = image.shape
    Y_sub, Cb_sub, Cr_sub = subsampling
    
    # subsample Y channel
    Y = image[:,:,0]
    Y_subsampled = np.zeros((height//Y_sub, width//Y_sub))
    for i in range(0, height, Y_sub):
        for j in range(0, width, Y_sub):
            Y_subsampled[i//Y_sub, j//Y_sub] = Y[i, j]
    
    # subsample Cb channel
    Cb = image[:,:,1]
    Cb_subsampled = np.zeros((height//Cb_sub, width//Cb_sub))
    for i in range(0, height, Cb_sub):
        for j in range(0, width, Cb_sub):
            Cb_subsampled[i//Cb_sub, j//Cb_sub] = Cb[i, j]
    
    # subsample Cr channel
    Cr = image[:,:,2]
    Cr_subsampled = np.zeros((height//Cr_sub, width//Cr_sub))
    for i in range(0, height, Cr_sub):
        for j in range(0, width, Cr_sub):
            Cr_subsampled[i//Cr_sub, j//Cr_sub] = Cr[i, j]
    
    return Y_subsampled, Cb_subsampled, Cr_subsampled


def upsample(subsampled, upsampling):
    # upsampling: tuple with 3 values indicating the upsampling ratio for Y, Cb and Cr
    Y_subsampled, Cb_subsampled, Cr_subsampled = subsampled
    Y_up, Cb_up, Cr_up = upsampling
    
    # upsample Y channel
    height, width = Y_subsampled.shape
    Y_upsampled = np.zeros((height*Y_up, width*Y_up))
    for i in range(height):
        for j in range(width):
            Y_upsampled[i*Y_up:(i+1)*Y_up, j*Y_up:(j+1)*Y_up] = Y_subsampled[i, j]
    
    # upsample Cb channel
    height, width = Cb_subsampled.shape
    Cb_upsampled = np.zeros((height*Cb_up, width*Cb_up))
    for i in range(height):
        for j in range(width):
            Cb_upsampled[i*Cb_up:(i+1)*Cb_up, j*Cb_up:(j+1)*Cb_up] = Cb_subsampled[i, j]
    
    # upsample Cr channel
    height, width = Cr_subsampled.shape
    Cr_upsampled = np.zeros((height*Cr_up, width*Cr_up))
    for i in range(height):
        for j in range(width):
            Cr_upsampled[i*Cr_up:(i+1)*Cr_up, j*Cr_up:(j+1)*Cr_up] = Cr_subsampled[i, j]
    
    return np.dstack((Y_upsampled, Cb_upsampled, Cr_upsampled))
