 #! /usr/bin python3
from roipoly import RoiPoly
import matplotlib
#matplotlib.use("TkAgg")
#from matplotlib import pyplot as plt 
import cv2
import os

if __name__ == '__main__':
    folder = "trainset"
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        #cv2.imshow('image', img)
        my_roi = RoiPoly(color='r')
        #cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





#Covarande methods
#--------------------
                tmp = np.zeros((3,3))
                #Set SigmaK 
                #First Class 
                for x in range(0,split[0]):
                    if Y[x] == 1:
                        p = (X[x] - mu[0]) 
                        p = p.reshape(1,3)
                        pre_trans = p.reshape(3,1)
                        T = np.transpose(pre_trans)
                        tmp = tmp + (p * T)
                sigmaK[0] = tmp/ split[0]
                tmp = np.zeros((3,3))
                # Next Class
                for x in range(split[0],split[1]):
                    if Y[x] == 2:
                        p = (X[x] - mu[1]) 
                        pre_trans = p.reshape(3,1)
                        T = np.transpose(pre_trans)
                        tmp += (p) * T 
                sigmaK[1] = tmp / (split[1] - split[0])
                tmp = np.zeros((3,3))
                # Next Class
                for x in range(split[1],split[2]):
                    if Y[x] == 3:
                        p = (X[x] - mu[2]) 
                        pre_trans = p.reshape(3,1)
                        T = np.transpose(pre_trans)
                        tmp += (p) * T 
                sigmaK[2] = tmp /(split[2] - split[1])

###-------------------
		tmp_x1x1,tmp_x1x2,tmp_x1x3 = 0,0,0
                tmp_x2x2,tmp_x2x3 = 0,0 
                tmp_x3x3 = 0 
                #Set SigmaK 
                #First Class 
                for x in range(0,split[0]):
                    if Y[x] == 1:
                        tmp = X[x] - mu[0] 
                        tmp_x1x1 += tmp[0]*tmp[0] 
                        tmp_x1x2 += tmp[0]*tmp[1] 
                        tmp_x1x3 += tmp[0]*tmp[2]
                        tmp_x2x2 += tmp[1]*tmp[1]
                        tmp_x2x3 += tmp[1]*tmp[2]
                        tmp_x3x3 += tmp[2]*tmp[2] 
                sigmaK[0,0,0] = tmp_x1x1 / (split[0]-1) 
                sigmaK[0,0,1] = tmp_x1x2 /(split[0] -1) 
                sigmaK[0,0,2] = tmp_x1x3 / (split[0] -1) 
                sigmaK[0,1,0] = sigmaK[0,0,1]
                sigmaK[0,1,1] = tmp_x2x2 / (split[0]-1)
                sigmaK[0,1,2] = tmp_x2x3 / (split[0] -1) 
                sigmaK[0,2,0] = sigmaK[0,0,2]
                sigmaK[0,2,1] = sigmaK[0,1,2] 
                sigmaK[0,2,2] = tmp_x3x3 / (split[0] -1) 
                # Next Class
                tmp_x1x1,tmp_x1x2,tmp_x1x3 = 0,0,0
                tmp_x2x2,tmp_x2x3 = 0,0 
                tmp_x3x3 = 0 
                for x in range(split[0],split[1]):
                    if Y[x] == 2:
                        tmp = X[x] - mu[1] 
                        tmp_x1x1 += tmp[0]*tmp[0] 
                        tmp_x1x2 += tmp[0]*tmp[1] 
                        tmp_x1x3 += tmp[0]*tmp[2]
                        tmp_x2x2 += tmp[1]*tmp[1]
                        tmp_x2x3 += tmp[1]*tmp[2]
                        tmp_x3x3 += tmp[2]*tmp[2] 
                sigmaK[1,0,0] = tmp_x1x1 / (split[1]- split[0]-1) 
                sigmaK[1,0,1] = tmp_x1x2 /(split[1] - split[0] -1) 
                sigmaK[1,0,2] = tmp_x1x3 / (split[1] - split[0] -1) 
                sigmaK[1,1,0] = sigmaK[1,0,1]
                sigmaK[1,1,1] = tmp_x2x2 / (split[1] - split[0]-1)
                sigmaK[1,1,2] = tmp_x2x3 / (split[1] - split[0] -1) 
                sigmaK[1,2,0] = sigmaK[1,0,2]
                sigmaK[1,2,1] = sigmaK[1,1,2] 
                sigmaK[1,2,2] = tmp_x3x3 / (split[1] - split[0] -1) 
                # Next Class
                tmp_x1x1,tmp_x1x2,tmp_x1x3 = 0,0,0
                tmp_x2x2,tmp_x2x3 = 0,0
                tmp_x3x3 = 0
                for x in range(split[1],split[2]):
                    if Y[x] == 3:
                        tmp = X[x] - mu[2]
                        tmp_x1x1 += tmp[0]*tmp[0]
                        tmp_x1x2 += tmp[0]*tmp[1]
                        tmp_x1x3 += tmp[0]*tmp[2]
                        tmp_x2x2 += tmp[1]*tmp[1]
                        tmp_x2x3 += tmp[1]*tmp[2]
                        tmp_x3x3 += tmp[2]*tmp[2]
                sigmaK[2,0,0] = tmp_x1x1 / (split[2]- split[1]-1)
                sigmaK[2,0,1] = tmp_x1x2 /(split[2] - split[1] -1)
                sigmaK[2,0,2] = tmp_x1x3 / (split[2] - split[1] -1)
                sigmaK[2,1,0] = sigmaK[2,0,1]
                sigmaK[2,1,1] = tmp_x2x2 / (split[2] - split[1]-1)
                sigmaK[2,1,2] = tmp_x2x3 / (split[2] - split[1] -1)
                sigmaK[2,2,0] = sigmaK[2,0,2]
                sigmaK[2,2,1] = sigmaK[2,1,2]
                sigmaK[2,2,2] = tmp_x3x3 / (split[2] - split[1] -1)
