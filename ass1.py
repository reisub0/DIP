#!/usr/bin/env python3

import numpy as np
import cv2
from pdb import set_trace

def downSample(image, k):
    return image[: : k, : : k]

def quantize(image, origLevels, newLevels):
    quantizedImage = image.flatten()
    factor = origLevels/newLevels
    for i,pix in enumerate(quantizedImage):
        quantizedImage[i] = int(pix / factor) * factor
    return quantizedImage.reshape(image.shape)

im1 = cv2.imread('lena.png')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
print(im1.shape)

downSamples = [ int(x) for x in input("Enter downsamples with spaces:").split()]
cv2.imshow('orig', im1)
for factor in downSamples:
    im2 = downSample(im1, factor)
    cv2.imshow('ds', im2)
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        continue

quantizers = [ int(x) for x in input("Enter quantizers with spaces:").split()]
cv2.imshow('orig', im1)
for quantizer in quantizers:
    im2 = quantize(im1, 256, quantizer)
    cv2.imshow('ds', im2)
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        continue
# im2 = quantize(im1, 256, 8)

# Display the resulting frame
while not (cv2.waitKey(0) & 0xFF == ord('q')):
    pass
# When everything done, release the capture
cv2.destroyAllWindows()
