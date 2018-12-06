# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:22:28 2018

@author: durus and jdlips
"""

import numpy as np
import cv2

class SeamCarver:

    def __init__(self, inputFilename, outputFilename, outputWidth, outputHeight):
        # Setting input parameters
        self.inputFilename = inputFilename
        self.inputImg = cv2.imread(inputFilename)
        self.inputHeight = np.size(self.inputImg, 0)
        self.inputWidth = np.size(self.inputImg, 1)

        # Setting output parameters
        self.outputFilename = outputFilename
        self.outputImg = np.copy(self.inputImg)
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight

    def seamCarving(self):
        colSeams = self.inputWidth - self.outputWidth
        # Checking if we are removing seams or adding them
        if colSeams > 0:
            self.removeSeams(colSeams)
        elif colSeams < 0:
            self.addSeams(-1 * colSeams)

    def removeSeams(self, seams):
        count = 0
        while count < seams:
            energyMap = self.getEnergyMap()
            energyValuesDown = self.getConsecutiveEnergyValues(energyMap)
            #leastEnergySeam = self.getLeastEnergySeam(energyValuesDown)
            #self.removeSeam(leastEnergySeam)
            count += 1

    def getEnergyMap(self):
        blue = self.outputImg[:,:,0]
        green = self.outputImg[:,:,1]
        red = self.outputImg[:,:,2]
        blueEnergy = np.absolute(cv2.Scharr(blue, -1, 1, 0)) + np.absolute(cv2.Scharr(blue, -1, 0, 1))
        greenEnergy = np.absolute(cv2.Scharr(green, -1, 1, 0)) + np.absolute(cv2.Scharr(green, -1, 0, 1))
        redEnergy = np.absolute(cv2.Scharr(red, -1, 1, 0)) + np.absolute(cv2.Scharr(red, -1, 0, 1))
        energyMap = blueEnergy + greenEnergy + redEnergy
        return energyMap

    def getConsecutiveEnergyValues(self, energyMap):
        energyValuesDown = np.copy(energyMap)
        rows, cols = energyMap.shape
        print(rows)
        print(self.inputHeight)
        currentCol = 0
        while (currentCol < self.inputWidth):
            currentRow = 0
            while (currentRow < self.inputHeight):
                #if (currentRow == 0):

                currentRow += 1
            currentCol += 1
        return 0



    #TODO: [X] Finish seamCarving(self):
        #TODO: [X] Finsh removeSeams(self, seams):
        #TODO: [X] Finish getEnergyMap(self):
            #TODO: [ ] Finish getConsecutiveEnergyValues(self, energyValues):
            #TODO: [ ] start getLeastEnergySeam(self, energyValuesDown):
            #TODO: [ ] start removeSeam(self, leastEnergySeam):
        #TODO: [ ] start addSeams():


    # Old Code
"""
    def __init__(self, img):
        self.img = img

    def min_seam(self):


#        Find the path of least energy from the top of the image to the bottom
#
#        Args:
#            self - SeamCarver instance
#        Returns:
#            mins - an np array that stores the minimum energy value that stores
#                   the minimum energy value seen so far
#            backtrack - contains the list of pixels in the seam



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
"""
