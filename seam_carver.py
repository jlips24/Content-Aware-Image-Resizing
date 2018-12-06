# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:22:28 2018

@author: durus
"""

import numpy as np
import cv2
#from utils import *

class SeamCarver:

    def __init__(self, inputFilename, outputFilename, outputWidth, outputHeight):
        # Setting input parameters
        self.inputFilename = inputFilename
        self.inputImg = cv2.imread(inputFilename)
        self.inputHeight = np.size(self.inputImg, 0)
        self.inputWidth = np.size(self.inputImg, 1)

        # Setting output parameters
        self.outputFilename = outputFilename
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight

    def min_seam(self):
        """
        Find the path of least energy from the top of the image to the bottom

        Args:
            self - SeamCarver instance
        Returns:
            mins - an np array that stores the minimum energy value that stores
                   the minimum energy value seen so far
            backtrack - contains the list of pixels in the seam
        """

        row, col, x = self.img.shape

        e_map = energy_map(self.img)
        mins = e_map.copy()
        backtrack = np.zeros(mins.shape, dtype=np.int)

        print("hi")

        for r in range(1, row):
            for c in range(0, col):
                if c == 0:
                    idx = np.argmin(mins[r - 1, c:c + 2])
                    backtrack[r, c] = idx + c
                    min_energy = mins[r - 1, idx + c]
                else:
                    idx = np.argmin(mins[r - 1, c - 1:c + 2])
                    backtrack[r, c] = idx + c - 1
                    print(idx)
                    print(c)
                    if idx + c - 1 >= 1428:
                        min_energy = mins[r - 1, col - 1]
                    else:
                        min_energy = mins[r - 1, idx + c - 1]

                mins[r, c] += min_energy

        return mins, backtrack
