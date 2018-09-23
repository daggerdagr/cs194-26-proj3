from scipy import signal
import skimage as sk
import skimage.filters as sf
import numpy as np

from enum import Enum

## custom
from utils import *

class Operation(Enum):
    Unsharp = "unsharp"
    GaussBlur = "gaussBlur"

def apply(operation, im1, im2):
    if operation == Operation.Unsharp:
        assert im2 == None
        return unsharpOp(im1)
    elif operation == Operation.GaussBlur:
        assert im2 == None
        return gaussBlurOp(im1)
    else:
        raise Exception("unrecognized operation : %s" % operation)

##################
# OPERATIONS
##################

def unsharpOp(im, alpha = 1.0, sigma = 10):
    gaussKernel = makeGaussKernel(sigma)

    unitImpulseKernel = signal.unit_impulse(gaussKernel.shape, idx="mid")

    totalKernel = unitImpulseKernel * (1 + alpha) - (gaussKernel * alpha)

    result = signal.convolve2d(im, totalKernel, mode="same")

    # gaussIm = signal.convolve2d(im, gaussKernel, mode="same")
    # testImage("testGauss.jpg", gaussIm)

    result = np.clip(result, -1, 1)

    testImage("testUnsharp.jpg", result)

def gaussBlurOp(im, sigma=10):
    gaussKernel = makeGaussKernel(sigma)

    result = signal.convolve2d(im, gaussKernel, mode="same")

    testImage("testGaussBlurOp.jpg", result)
    return result


# returns a Gauss kernel -- desired MxM size MUST have M = an odd number
def makeGaussKernel(lowSig = 1, kernelSize = None):
    if kernelSize == None:
        kernelSize = lowSig * 3
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

def hybridImageOp(im1, im2, sigma1, sigma2):

    # im1 = derek -- cut out the high signals
    # im2 = cat -- cut out low pass



    return