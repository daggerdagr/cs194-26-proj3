import skimage as sk
import skimage.filters as sf
import numpy as np

from enum import Enum

## custom
from utils import *

class Operation(Enum):
    Unsharp = "unsharp"

def apply(operation, im1, im2):
    if operation == Operation.Unsharp:
        assert im2 == None
        return unsharpOp(im1)
    else:
        raise Exception("unrecognized operation : %s" % operation)

##################
# OPERATIONS
##################

def unsharpOp(im):
    gaussIm = sf.gaussian(im, 10)
    testImage("testGauss.jpg", gaussIm)
    result = im + (im - gaussIm)
    result = np.clip(result, -1, 1)

    testImage("testUnsharp.jpg", result)

# returns a Gauss kernel -- desired MxM size MUST have M = an odd number
def makeGaussKernel(kernelSize, lowSig = 1):
    assert kernelSize > 0
    assert kernelSize % 2 != 0

    absEdgeVal = kernelSize // 2

    result = np.empty([kernelSize, kernelSize])

    for preU in range(kernelSize):
        u = preU - absEdgeVal
        for preV in range(kernelSize):
            v = preV - absEdgeVal
            h = 1 / (2 * np.pi * (lowSig ** 2)) * np.exp(-1 * (u ** 2 + v ** 2) / (lowSig ** 2))
            result[preU, preV] = h

    return result
