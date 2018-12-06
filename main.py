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
    outputWidth = 1427
    outputHeight = 968
    outputFilename = "output/castle_" + str(outputWidth) + "x" + str(outputHeight) + ".jpg"


    sC = SeamCarver(inputFilename, outputFilename, outputWidth, outputHeight, True)
    sC.seamCarving();
    cv2.imwrite(outputFilename, sC.outputImg)
    print("Done")
    #energyMap = sC.getEnergyMap()
    #mins, backtrack = sC.getCumulativeMaps(energyMap)
    #out = sC.getLeastEnergySeam(mins)
    #sC.removeSeam
    #imwrite(sC.outputImg(Out))
    #print(out.shape)
    #print(out)

if __name__ == "__main__":
    main()
