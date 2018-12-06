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
            energyValuesDown = self.getCumulativeMaps(energyMap)
            leastEnergySeam = self.getLeastEnergySeam(energyValuesDown[0])
            self.removeSeam(leastEnergySeam)
            count += 1

    def getEnergyMap(self):

        blue, green, red = self.split_channels()

        blueEnergy = np.absolute(cv2.Scharr(blue, -1, 1, 0)) + np.absolute(cv2.Scharr(blue, -1, 0, 1))
        greenEnergy = np.absolute(cv2.Scharr(green, -1, 1, 0)) + np.absolute(cv2.Scharr(green, -1, 0, 1))
        redEnergy = np.absolute(cv2.Scharr(red, -1, 1, 0)) + np.absolute(cv2.Scharr(red, -1, 0, 1))
        energyMap = blueEnergy + greenEnergy + redEnergy
        return energyMap
        """
        scharrx = cv2.Scharr(self.outputImg, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(self.outputImg, cv2.CV_64F, 0, 1)

        abs_grad_x = cv2.convertScaleAbs(scharrx)
        abs_grad_y = cv2.convertScaleAbs(scharry)


        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
        return grad
        """

    def getCumulativeMaps(self, energyMap):

        blue, green, red = self.split_channels()

        xKernel = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]])
        yLeftKernel = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        yRightKernel = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]])

        xNeighbors = np.absolute(cv2.filter2D(blue, -1, kernel=xKernel)) + \
                 np.absolute(cv2.filter2D(green, -1, kernel=xKernel)) + \
                 np.absolute(cv2.filter2D(red, -1, kernel=xKernel))

        yLeft = np.absolute(cv2.filter2D(blue, -1, kernel=yLeftKernel)) + \
                 np.absolute(cv2.filter2D(green, -1, kernel=yLeftKernel)) + \
                 np.absolute(cv2.filter2D(red, -1, kernel=yLeftKernel))

        yRight = np.absolute(cv2.filter2D(blue, -1, kernel=yRightKernel)) + \
                 np.absolute(cv2.filter2D(green, -1, kernel=yRightKernel)) + \
                 np.absolute(cv2.filter2D(red, -1, kernel=yRightKernel))

        mins = np.copy(energyMap)
        backtrack = np.copy(energyMap)
        row, col = energyMap.shape

        for r in range(1, row):
            for c in range(0, col):
                if c == 0:
                    mins[r, c] = energyMap[r, c] + min(mins[r - 1, c + 1] + xNeighbors[r - 1, c + 1] + \
                                                        yRight[r - 1, c + 1],
                                                        mins[r - 1, c] + xNeighbors[r - 1, c])
                elif c == col - 1:
                    mins[r, c] = energyMap[r, c] + min(mins[r - 1, c - 1] + xNeighbors[r - 1, c - 1] + \
                                                        yLeft[r - 1, c - 1],
                                                        mins[r - 1, c] + xNeighbors[r - 1, c])
                else:
                    mins[r, c] = energyMap[r, c] + min(mins[r - 1, c - 1] + xNeighbors[r - 1, c - 1] + \
                                                        yLeft[r - 1, c - 1],
                                                        mins[r - 1, c + 1] + xNeighbors[r - 1, c + 1] + \
                                                        yRight[r - 1, c + 1],
                                                        mins[r - 1, c] + xNeighbors[r - 1, c])
                backtrack[r, c] = energyMap[r, c] + np.amin(backtrack[r - 1, max(c - 1, 0): min(c + 2, col - 1)])

        return mins, backtrack

    def getLeastEnergySeam(self, energyValuesDown):
        m, n = energyValuesDown.shape
        lis = np.zeros((m,), dtype=np.uint32)
        lis[-1] = np.argmin(energyValuesDown[-1])
        for row in range(m - 2, -1, -1):
            prv_x = lis[row + 1]
            if prv_x == 0:
                lis[row] = np.argmin(energyValuesDown[row, : 2])
            else:
                lis[row] = np.argmin(energyValuesDown[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return lis

    def split_channels(self):

        blue = self.outputImg[:,:,0]
        green = self.outputImg[:,:,1]
        red = self.outputImg[:,:,2]

        return blue, green, red

    def removeSeam(self, leastEnergySeam):
        row, col = self.outputImg.shape[: 2]
        output = np.zeros((row, col - 1, 3))
        for r in range(row):
            c = leastEnergySeam[r]
            for i in range(3):
                output[r, :, i] = np.delete(self.outputImg[r, :, i], [c])
        self.outputImg = np.copy(output)

    #TODO: [X] Finish seamCarving(self):
        #TODO: [X] Finsh removeSeams(self, seams):
        #TODO: [X] Finish getEnergyMap(self):
            #TODO: [X] Finish getCumulativeMaps(self, energyValues):
                #TODO: [ ]Test getCumulativeMaps(self, energyMap)
            #TODO: [X] start getLeastEnergySeam(self, energyValuesDown):
                #TODO: [ ] change getLeastEnergySeam more
            #TODO: [ ] start removeSeam(self, leastEnergySeam):
        #TODO: [ ] start addSeams():
    #TODO: [ ] Test with multile images and output sizes
