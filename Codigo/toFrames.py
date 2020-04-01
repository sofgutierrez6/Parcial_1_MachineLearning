# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:39:10 2020

@author: Daniel Guatibonza
"""


import cv2
vidcap = cv2.VideoCapture('SuperMarioBros.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.png" % count, image)  
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1