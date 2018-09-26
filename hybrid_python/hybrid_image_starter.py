import matplotlib.pyplot as plt
from hybrid_python.align_image_code import align_images
from operations import hybrid_image, pyramidsOp, PyramidMode
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

printImage("save.jpg", hybrid, False)

# plt.imshow(hybrid)
# plt.show

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
# stack = pyramidsOp(im1_aligned, N, 4, mode=PyramidMode.Laplacian)
stack = pyramidsOp(hybrid, N, 1, mode=PyramidMode.Laplacian)

for i in range(len(stack)):
    printImage("pyr_"+str(i)+".bmp", stack[i], disp=False)