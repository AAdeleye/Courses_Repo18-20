import os, cv2
from skimage.measure import label, regionprops
import numpy as np  



def samplePlacement(train,Samp):
    '''         Load pixels as samples
    '''
    for img in train:
        rows = img.shape[0]
        cols = img.shape[1]
        for i in range(0,rows):
            for j in range(0,cols):
                Samp.append(img[i,j])   
def weights():
    #Weights 
    ''' The catogory of weights are Blue,NoneBlue,Grey'''
    theta  = np.zeros(3) # A scalr probabilty for each class
    mu = np.zeros((3,3)) # A 1x3 vector of each class
    sigma = np.zeros((3,3,3)) # A 3x3 matrix of 

    #Pre-examples 
    train_img =  [ ] 
    #Samples (pixels) 
    X = [ ] 
    #Point where lable changes
    split = [ ] 

    folder = "Blue_bin_clips"
    for filename in os.listdir(folder):
        if filename[0] == '.':
            continue 
        img = cv2.imread(os.path.join(folder,filename)) 
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
        train_img.append(hsv_img)

    samplePlacement(train_img,X)
    split.append(len(X))
    train_img.clear()

    folder = "Not_blue_clips"
    for filename in os.listdir(folder):
        if filename[0] == '.':
            continue 
        img = cv2.imread(os.path.join(folder,filename)) 
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
        train_img.append(hsv_img)

    samplePlacement(train_img,X)
    split.append(len(X))
    train_img.clear()

    folder = "Grey_clips"
    for filename in os.listdir(folder):
        if filename[0] == '.':
            continue 
        img = cv2.imread(os.path.join(folder,filename)) 
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
        train_img.append(hsv_img)

    samplePlacement(train_img,X)
    split.append(len(X))
    train_img.clear()

    # init Lables 
    Y = np.zeros(len(X)) 
    Y[0:split[0]] = 1   #Blue Bin
    Y[split[0]:split[1]] = 2 #None Blue 
    Y[split[1]:split[2]] = 3 #Grey 
    X = np.array(X)

    #Set Theta
    tmp_B, tmp_N, tmp_G = 0,0,0
    for x in range(0,len(X)):
        if Y[x] == 1:
            tmp_B += 1
        elif Y[x] == 2:
            tmp_N += 1 
        else: 
            tmp_G += 1
    theta[0] = tmp_B / len(X)
    theta[1] = tmp_N/ len(X)
    theta[2] = tmp_G/ len(X) 

    #Set Mu
    tmp_B, tmp_N, tmp_G =  np.zeros(3), np.zeros(3), np.zeros(3)
    tmpB,tmpN,tmpG = 0,0,0 
    for x in range(0,len(X)):
        if Y[x] == 1:
            tmp_B += X[x]
            tmpB += 1
        elif Y[x] == 2:
            tmp_N += X[x]
            tmpN += 1
        else:
            tmp_G += X[x]
            tmpG += 1
    mu[0] = tmp_B /tmpB 
    mu[1] = tmp_N/tmpN
    mu[2] = tmp_G/tmpG

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
    sigma[0,0,0] = tmp_x1x1 / (split[0]-1) 
    sigma[0,0,1] = tmp_x1x2 /(split[0] -1) 
    sigma[0,0,2] = tmp_x1x3 / (split[0] -1)
    sigma[0,1,0] = sigma[0,0,1]
    sigma[0,1,1] = tmp_x2x2 / (split[0]-1)
    sigma[0,1,2] = tmp_x2x3 / (split[0] -1)
    sigma[0,2,0] = sigma[0,0,2]
    sigma[0,2,1] = sigma[0,1,2] 
    sigma[0,2,2] = tmp_x3x3 / (split[0] -1)
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
    sigma[1,0,0] = tmp_x1x1 / (split[1]- split[0]-1) 
    sigma[1,0,1] = tmp_x1x2 /(split[1] - split[0] -1) 
    sigma[1,0,2] = tmp_x1x3 / (split[1] - split[0] -1)
    sigma[1,1,0] = sigma[1,0,1]
    sigma[1,1,1] = tmp_x2x2 / (split[1] - split[0]-1)
    sigma[1,1,2] = tmp_x2x3 / (split[1] - split[0] -1)
    sigma[1,2,0] = sigma[1,0,2]
    sigma[1,2,1] = sigma[1,1,2] 
    sigma[1,2,2] = tmp_x3x3 / (split[1] - split[0] -1)
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
    sigma[2,0,0] = tmp_x1x1 / (split[2]- split[1]-1) 
    sigma[2,0,1] = tmp_x1x2 /(split[2] - split[1] -1) 
    sigma[2,0,2] = tmp_x1x3 / (split[2] - split[1] -1)
    sigma[2,1,0] = sigma[2,0,1]
    sigma[2,1,1] = tmp_x2x2 / (split[2] - split[1]-1)
    sigma[2,1,2] = tmp_x2x3 / (split[2] - split[1] -1)
    sigma[2,2,0] = sigma[2,0,2]
    sigma[2,2,1] = sigma[2,1,2] 
    sigma[2,2,2] = tmp_x3x3 / (split[2] - split[1] -1)

    print(theta)
    print("mu",mu)
    print("sigma",sigma) 
    
    np.save('theta',theta)
    np.save('mu',mu) 
    np.save('sigma',sigma)

if __name__ == '__main__':
    weights()  

  
  
  
  
  
