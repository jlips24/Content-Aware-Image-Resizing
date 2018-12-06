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

    def removeSeams(seams):
        count = 0
        while count < seams:
            energyValues = self.getEnergyValues()
            #energyValuesDown = self.getEnergyValuesDown(energyValues)
            #leastEnergySeam = self.getLeastEnergySeam(energyValuesDown)
            #self.removeSeam(leastEnergySeam)
            count += 1

    def getEnergyValues():
        blue, green, red = cv2.split(self.outputImage)
        blueEnergy = np.absolute(cv2.Scharr(blue, -1, 1, 0)) + np.absolute(cv2.Scharr(blue, -1, 0, 1))
        greenEnergy = np.absolute(cv2.Scharr(green, -1, 1, 0)) + np.absolute(cv2.Scharr(green, -1, 0, 1))
        redEnergy = np.absolute(cv2.Scharr(red, -1, 1, 0)) + np.absolute(cv2.Scharr(red, -1, 0, 1))
        totalEnergy = blueEnergy + greenEnergy + redEnergy
        return totalEnergy

    #TODO: def getEnergyValuesDown(energyValues):
    #TODO: getLeastEnergySeam(energyValuesDown):
    #TODO: removeSeam(leastEnergySeam):
