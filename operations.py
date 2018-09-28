from scipy import signal
import skimage as sk
import skimage.filters as sf
import numpy as np

from enum import Enum

## custom
from utils import *

# class Operation(Enum):
#     #1 Arg
#     Unsharp = unsharpOp
#     GaussBlur_1D = gaussBlurOp_1D
#     GaussBlur_3D = gaussBlurOp_3D
#
#     arg1 = set(Unsharp, GaussBlur_1D, GaussBlur_3D)
#
# def apply(operation, im1, im2):
#     if operation in Operation.arg1:
#         assert im2 == None
#         return operation(im1)
#     elif operation == Operation.GaussBlur_1D:
#         assert im2 == None
#         return gaussBlurOp_1D(im1)
#     elif operation == Operation.GaussBlur_3D:
#         assert im2 == None
#         return gaussBlurOp_3D(im1)
#     else:
#         raise Exception("unrecognized operation : %s" % operation)

##################
# OPERATIONS
##################

def unsharpOp(im, alpha = 1.0, sigma = 10):
    gaussKernel = makeGaussKernel(sigma)

    unitImpulseKernel = signal.unit_impulse(gaussKernel.shape, idx="mid")

    totalKernel = unitImpulseKernel * (1 + alpha) - (gaussKernel * alpha)

    result = np.zeros(im.shape)

    for i in range(3):
        result[:, :, i] = signal.convolve2d(im[:, :, i], totalKernel, mode="same")

    # gaussIm = signal.convolve2d(im, gaussKernel, mode="same")
    # testImage("testGauss.jpg", gaussIm)

    result = np.clip(result, -1, 1)

    # testImage("testUnsharp.jpg", result)
    return result

def gaussBlurOp_1D(im, sigma=10):
    gaussKernel = makeGaussKernel(sigma)

    result = signal.convolve2d(im, gaussKernel, mode="same", boundary="symm")

    # testImage("testGaussBlurOp.jpg", result)
    return result

def gaussBlurOp_3D(im, sigma=10):
    assert im.ndim == 3

    result = []

    for i in range(3):
        result.append(gaussBlurOp_1D(im[:, :, i], sigma))

    return np.dstack(result)

# returns a Gauss kernel -- desired MxM size MUST have M = an odd number
def makeGaussKernel(lowSig = 1, kernelSize = None):
    if kernelSize == None:
        kernelSize = int(np.ceil(lowSig * 3))
    assert kernelSize > 0
    # assert kernelSize % 2 != 0

    absEdgeVal = kernelSize // 2

    result = np.empty([kernelSize, kernelSize])

    sum = 0
    for preU in range(kernelSize):
        u = preU - absEdgeVal
        for preV in range(kernelSize):
            v = preV - absEdgeVal
            h = 1 / (2 * np.pi * (lowSig ** 2)) * np.exp(-1 * (u ** 2 + v ** 2) / (lowSig ** 2))
            result[preU, preV] = h
            sum += h

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            result[y, x] /= sum

    return result

def hybrid_image(im1, im2, sigma1, sigma2):
    return hybridImageOp(im1, im2, sigma1, sigma2)

def hybridImageMidstep(im, sigma):
    # image - gauss blurred image
    gaussKernel = makeGaussKernel(lowSig=sigma)
    unitImpulseKernel = signal.unit_impulse(gaussKernel.shape, idx="mid")

    totalKernel = unitImpulseKernel - gaussKernel
    # totalKernel = gaussKernel

    result = signal.convolve2d(im, totalKernel, mode="same")
    return result

def hybridImageOp(im1, im2, sigma1, sigma2):

    # im1 = derek -- cut out the high signals
    # im2 = cat -- cut out low pass

    # viewImage(im1)
    # viewImage(im2)

    res = np.zeros(im1.shape)

    for i in range(3):
        firstIm = im1[:, :, i]
        secondIm = im2[:, :, i]
        fin1 = gaussBlurOp_1D(firstIm, sigma1)
        # viewImage(fin1)
        fin2 = secondIm - gaussBlurOp_1D(secondIm, sigma2)
        # fin2a = gaussBlurOp_1D(secondIm, sigma2)
        # fin2a = hybridImageMidstep(secondIm, sigma2)
        # assert np.array_equal(fin2a, fin2)
        # viewImage(fin2)
        res[:, :, i] = np.dot(fin1 + fin2, 1/2)

    return res


#### PYRAMID

class PyramidMode(Enum):
    Gaussian = "gauss"
    Laplacian = "laplacian"

def pyramidsOp(im, levels, sigma, mode = PyramidMode.Gaussian):
    if mode == PyramidMode.Gaussian:
        return gaussStackOp_3D(im, levels, sigma)
    elif mode == PyramidMode.Laplacian:
        return laplacianPyrOp_3D(im, levels, sigma)

def gaussStackOp_3D(im, levels, sigma):
    assert levels > 0
    #inclusive of original img, at layer indexed 0

    result = []
    # newLayer = (lambda: np.zeros(im.shape))

    for i in range(levels+1):
        if i == 0:
            result.append(im)
            continue
        # currLayer = newLayer()
        currLayer = gaussBlurOp_3D(result[i-1], sigma)
        result.append(currLayer)

    return np.array(result)

def laplacianPyrOp_3D(im, levels, sigma, scaleB = False):
    gaussStack = gaussStackOp_3D(im, levels, sigma)

    for i in range(levels):
        res = gaussStack[i] - gaussStack[i+1]
        if scaleB:
            finalCurrLayer = (res - res.min()) / (res.max() - res.min())
        else:
            finalCurrLayer = res
        gaussStack[i] = finalCurrLayer

    return gaussStack


def scaler(LM):  # scales to 0 1
    return np.dot(LM - LM.min(), 1 / (LM.max() - LM.min()))  # * 2 - 1


def multiResBlendOp(im1, im2, mask, levels, sigma):
    assert im1.shape == im2.shape == mask.shape

    L1 = laplacianPyrOp_3D(im1, levels, sigma)
    L2 = laplacianPyrOp_3D(im2, levels, sigma)
    LM = gaussStackOp_3D(mask, levels, sigma)  # laplacianPyrOp_3D(mask, levels, sigma)

    # LM_blur = gaussStackOp_3D(mask, levels, sigma)

    # for i in range(len(L1)):
    #     viewImage(LM[i])

    #     LM1 = np.dot(LM - LM.min(), 1 / (LM.max() - LM.min())) ## scale LM to [0, 1]
    #     LM1 = LM1

    #     print(LM.min(), LM.max())
    LM1 = LM
    LM2 = (1 - LM1)

    #     print("HELP: ", LM1[0, 0, 0, 0], LM2[0, 0, 0, 0])
    #     print(L1.min(), L1.max())
    #     print(L2.min(), L2.max())
    #     return

    L1_post = LM1 * L1
    # for i in range(len(L1)):
    #     viewImage(scaler(L1_post[i]))
    L2_post = LM2 * L2

    finalL = L1_post + L2_post

    # levelsDim, heightDim, widthDim, channelDim = L1.shape

    tes = np.zeros(L1[0].shape)

    for i in range(len(L1)):
        # viewImage(scaler(finalL[i]))
        tes += finalL[i]

    return tes