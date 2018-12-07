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
    outputWidth = 900
    outputHeight = 900
    outputFilename = "output/castle_" + str(outputWidth) + "x" + str(outputHeight) + ".jpg"


    sC = SeamCarver(inputFilename, outputFilename, outputWidth, outputHeight, True)
    sC.seamCarving();
    sC.outputImageToFile(outputFilename, sC.outputImg)
    print("Done")

if __name__ == "__main__":
    main()
