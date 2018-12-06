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
    outputWidth = 968
    outputHeight = 968
    outputFilename = "output/castle_" + str(outputWidth) + "x" + str(outputHeight) + ".jpg"


    sC = SeamCarver(inputFilename, outputFilename, outputWidth, outputHeight)
    sC.seamCarving()

if __name__ == "__main__":
    main()
