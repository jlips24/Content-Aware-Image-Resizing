# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:53:08 2018

@author: durus
"""

import numpy as np
import cv2

def energy_map(img):
    
    window_name = ('Sobel Demo - Simple Edge Detector')
    
    sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)
       
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    cv2.imshow(window_name, grad)
    cv2.waitKey(0)
    
    return grad
            
def main():
    img = cv2.imread('castle.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('Image', img)
    energy_map(img)
            
if __name__ == '__main__':
    main()
