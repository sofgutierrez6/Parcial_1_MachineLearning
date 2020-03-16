import Image, os            
import numpy as np                
import matplotlib.pyplot as plt


N_images = 1000

for i in range(N_images):
    I = Image.open(os.getcwd() + "\\" +i+".jpeg")
    print(I)