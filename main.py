"""
Created on Wed Dec 5 22:44:37 2018

@author: jdlips
"""

import cv2
import numpy as np
from seam_carver import SeamCarver
import os

def main():
    inputFilename = "input/castle.jpg"
    outputFilename = "output/castle_carved.jpg"
    outputWidth = 968
    outputHeight = 968

    inputImg = cv2.imread(inputFilename)

    inputWidth = np.size(inputImg, 0)

    sC = SeamCarver(inputFilename, outputFilename, outputWidth, outputHeight)

if __name__ == "__main__":
    main()
