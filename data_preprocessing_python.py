
# coding: utf-8

# # Data pre-processing steps
# 
# This notebook will read the svs files in the data file using openslide python and apply the following pre-processing steps:
# 
# -Crop the images in small batches of size 896*896
# 
# -Filter out the tiles that contain less than 90% of brain (Tumor??) tissue: Hysteresis thresholding on the grayscale and 8-bit depth complemented image (http://ac.els-cdn.com/S1361841515001838/1-s2.0-S1361841515001838-main.pdf?_tid=f96cb1fa-35ba-11e7-b61d-00000aab0f6b&acdnat=1494446462_d2ee895640e38bd660bc559fc6233d34)
# 
# -Optional: Nuclei segmentation using morphometric top-hat filtering and hysteresis thresholding (http://ac.els-cdn.com/S1361841515001838/1-s2.0-S1361841515001838-main.pdf?_tid=f96cb1fa-35ba-11e7-b61d-00000aab0f6b&acdnat=1494446462_d2ee895640e38bd660bc559fc6233d34)
# 
# -Further tile to 224*224 (input size of a ResNet or Inception CNN)
# 
# input_size $= N*224*224*(1 \ or \ 3)$
# 
# label_size $= N$
# 
# N = Number_of_images * Number_of_patches
# 
# For now, the notebook only treats one svs file : "test.svs". The pre-processing steps are very computationnally expensive so we need to parallelize the code.

# In[41]:

from openslide import *
import numpy as np
import skimage
import matplotlib.pyplot as plt
import os

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from PIL import Image
from skimage import feature

import scipy.ndimage as ndi
from scipy.ndimage import (gaussian_filter,
                           generate_binary_structure, binary_erosion, label)

import matplotlib.image as mpimg


# modified from Canny edge detection algo
# https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py#L53
def canny_hyst(image, low_threshold, high_threshold):
    #
    # The steps involved:
    #
    # * Label all points above the high threshold as edges.
    # * Recursively label any point above the low threshold that is 8-connected
    #   to a labeled point as an edge.
    #
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    #

    #
    #---- Create two masks at the two thresholds.
    #
    high_mask = (image >= high_threshold)
    low_mask = (image >= low_threshold)
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                             np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask

def greyscale(img):
    # http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    # Method #1: PIL conversion
    #pil_greyscale = Image.fromarray(img, 'RGB').convert(mode = 'L')
    #pil_greyscale = np.array(pil_greyscale.getdata()).reshape((H, W,))
    #print ("PIL conversion")
    #plt.imshow(pil_greyscale.astype('uint8'), cmap='gray')
    #plt.show()

    # Method #2: LUMA coding
    luma_greyscale = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return luma_greyscale

def autocontrast(grey_img):
    H, W = grey_img.shape
    # Translated Divakar's Matlab code
    # https://www.mathworks.com/matlabcentral/fileexchange/10566-auto-contrast
    low_limit=0.008
    up_limit=0.992
    grey_img_flat = np.sort(grey_img.reshape(H * W))
    #print (grey_img_flat[int(np.ceil(low_limit*H*W))], grey_img_flat[int(np.ceil(up_limit*H*W))])
    v_min = grey_img_flat[int(np.ceil(low_limit*H*W))]
    v_max = grey_img_flat[int(np.ceil(up_limit*H*W))]
    grey_img = (grey_img-v_min)/(v_max-v_min)
    grey_img *= 255
    #Do we need a reshape here?
    return grey_img

def complement(grey_img):
    # Performing 8-bit complement
    # http://micro.magnet.fsu.edu/primer/java/digitalimaging/processing/complementimage/
    return 255 - grey_img
    
def filtering (img, debug = False):
    if debug:
        print ("Original Image")
        plt.imshow(img.astype('uint8'))
        plt.show()
    
    H, W, C = img.shape

    # Grayscale
    img = greyscale(img)
    if debug:
        print ("LUMA coding")
        plt.imshow(img.astype('uint8'), cmap='gray')
        plt.show()

    # TODO: This doesnt give us the results we are looking for
    # Autocontrast
    #img = autocontrast(img)
    #if debug:
    #    print ("Auto Contrast")
    #    plt.imshow(img.astype('uint8'), cmap='gray')
    #    plt.show()

    # Performing 8-bit complement
    # http://micro.magnet.fsu.edu/primer/java/digitalimaging/processing/complementimage/
    img = complement(img)
    if debug:
        print ("8-bit Complement")
        plt.imshow(img.astype('uint8'), cmap='gray')
        plt.show()

    # Hysteresis Thresholding
    img = canny_hyst(img, 50, 100)
    if debug:
        print ("Hysteresis 2")
        plt.imshow(img.astype('uint8'), cmap='gray')
        plt.show()

    if debug:
        print (float(np.sum(img)))
        print (float(np.sum(img)) / (H*W))
    return float(np.sum(img)) / (H*W)

def generate_valid_patches(filename, patch_size=896):
    img = OpenSlide("/home/cedoz/radiogenomics/data/%s"%filename)
    width, height = img.dimensions
    
    print("building and filtering patches...")
    X_train = np.zeros((1, patch_size, patch_size, 3))
    for i in range(int(height/patch_size)):
        print ("iteration %d out of %d"%(i,int(height/patch_size)))
        for j in range(int(width/patch_size)):
            idx = i*int(width/patch_size) + j
            patch = img.read_region(location = (j*patch_size,i*patch_size), level = 0, size = (patch_size,patch_size))
            patch = np.array(patch.getdata())[:,0:-1].reshape((patch_size, patch_size, 3))
            if filtering(patch) >= 0.5:
                patch = np.expand_dims(patch, axis = 0)
                X_train = np.append(X_train, patch, axis = 0)
    X_train = X_train[1:]
    return X_train

