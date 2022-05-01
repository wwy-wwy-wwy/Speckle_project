import pandas as pd
import numpy as np
import cv2
import csv
from skimage import io

def dedrift_videos(imgs,dedrift_file,foldername, imgname):
    data= pd.read_csv(dedrift_file)
    original=[data.XM[0],data.YM[0]]
    initial_area=data.Area[0]
    for index in range(0,imgs.shape[0],1):
        x_difference=original[0]-data.XM[index]
        y_difference=original[1]-data.YM[index]
        M = np.float32([[1, 0, x_difference], [0, 1, y_difference]])
        try:
            # Read image from disk.
            img = imgs[index]
            #cv2.imread(FILE_NAME)
            (rows, cols) = img.shape[:2]
  
            # warpAffine does appropriate shifting given the
            # translation matrix.
            res = cv2.warpAffine(img, M, (cols, rows))
            imgs[index]=res
        except IOError:
            print ('Error while reading files !!!')
    print(imgs.shape)
    io.imsave(foldername+imgname+'_dedrifted.tif',imgs)