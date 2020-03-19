from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2

N = 28
for i in range(1,N+1):
    image = io.imread(str(i)+".jpg")/255.0
    image = image[0:738]
    image = image[43:738]
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    io.imsave("data" + str(2877+i) + ".png", resized)