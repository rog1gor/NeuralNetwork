import numpy as np
import matplotlib.image as mpimg

neuron = float

#& Classifies a RGB pixel either as an empty pixel or a filled pixel
#& Returns:
#& 0. - empty pixel
#& 1. - filled pixel
def _classify_pixel(pixel: np.ndarray) -> neuron:
    assert pixel.shape == (3,), f"""
        Unexpected pixel shape.
        Expected shape: (3,)
        Actuall shape: {pixel.shape}"""
    
    average_rgb_val = np.sum(pixel) / len(pixel)
    if average_rgb_val < 0.75:
        return 1.
    return 0.

#& Transforms and image to a vector of "True/False" (1. / 0.) vector 
def img_to_neurons(png_file: str) -> np.ndarray[neuron]:
    img = mpimg.imread(png_file)[:,:,:3]   #? Narrow to RGB only
    
    #? Reshape to get a vector of RGB pixels
    pixels = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    #? Convert each RGB pixel into a "black or white" pixel
    return np.apply_along_axis(_classify_pixel, axis=1, arr=pixels)
    

if __name__ == "__main__":
    print(img_to_neurons("../training_data/to.png"))