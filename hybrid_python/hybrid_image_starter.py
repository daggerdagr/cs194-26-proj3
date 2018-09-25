import matplotlib.pyplot as plt
from hybrid_python.align_image_code import align_images
from operations import hybrid_image, pyramids
from utils import *

# First load images

# high sf
im1 = plt.imread('./DerekPicture.jpg')/255.

# low sf
im2 = plt.imread('./nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 20
sigma2 = 21
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

printImage("save.jpg", hybrid)

# plt.imshow(hybrid)
# plt.show

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
pyramids(hybrid, N)