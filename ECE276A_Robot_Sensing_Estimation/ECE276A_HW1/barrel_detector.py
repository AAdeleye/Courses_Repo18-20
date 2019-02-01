'''
ECE276A WI19 HW1
Blue Barrel Detector
Akanimoh Adeleye
'''

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

class BarrelDetector():
        #Weights 
        ''' The catogory of weights are Blue,NoneBlue,Grey'''
        theta  = np.zeros(3) # A scalr probabilty for each class
        mu = np.zeros((3,3)) # A 1x3 vector of each class
        sigmaK = np.zeros((3,3,3)) # A 3x3 matrix of 
	
        def __init__(self):
                '''
                        Initilize your blue barrel detector with the attributes you need
                        eg. parameters of your classifier
                '''
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
                self.theta[0] = tmp_B / len(X)
                self.theta[1] = tmp_N/ len(X)
                self.theta[2] = tmp_G/ len(X) 
                
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
                self.mu[0] = tmp_B /tmpB 
                self.mu[1] = tmp_N/tmpN
                self.mu[2] = tmp_G/tmpG
                
                tmp_x1x1,tmp_x1x2,tmp_x1x3 = 0,0,0
                tmp_x2x2,tmp_x2x3 = 0,0
                tmp_x3x3 = 0
                #Set SigmaK 
                #First Class 
                for x in range(0,split[0]):
                    if Y[x] == 1:
                        tmp = X[x] - self.mu[0] 
                        tmp_x1x1 += tmp[0]*tmp[0] 
                        tmp_x1x2 += tmp[0]*tmp[1] 
                        tmp_x1x3 += tmp[0]*tmp[2]
                        tmp_x2x2 += tmp[1]*tmp[1]
                        tmp_x2x3 += tmp[1]*tmp[2]
                        tmp_x3x3 += tmp[2]*tmp[2] 
                self.sigmaK[0,0,0] = tmp_x1x1 / (split[0]-1) 
                self.sigmaK[0,0,1] = tmp_x1x2 /(split[0] -1) 
                self.sigmaK[0,0,2] = tmp_x1x3 / (split[0] -1)
                self.sigmaK[0,1,0] = self.sigmaK[0,0,1]
                self.sigmaK[0,1,1] = tmp_x2x2 / (split[0]-1)
                self.sigmaK[0,1,2] = tmp_x2x3 / (split[0] -1)
                self.sigmaK[0,2,0] = self.sigmaK[0,0,2]
                self.sigmaK[0,2,1] = self.sigmaK[0,1,2] 
                self.sigmaK[0,2,2] = tmp_x3x3 / (split[0] -1)
                # Next Class
                tmp_x1x1,tmp_x1x2,tmp_x1x3 = 0,0,0
                tmp_x2x2,tmp_x2x3 = 0,0
                tmp_x3x3 = 0
                for x in range(split[0],split[1]):
                    if Y[x] == 2:
                        tmp = X[x] - self.mu[1] 
                        tmp_x1x1 += tmp[0]*tmp[0] 
                        tmp_x1x2 += tmp[0]*tmp[1] 
                        tmp_x1x3 += tmp[0]*tmp[2]
                        tmp_x2x2 += tmp[1]*tmp[1]
                        tmp_x2x3 += tmp[1]*tmp[2]
                        tmp_x3x3 += tmp[2]*tmp[2] 
                self.sigmaK[1,0,0] = tmp_x1x1 / (split[1]- split[0]-1) 
                self.sigmaK[1,0,1] = tmp_x1x2 /(split[1] - split[0] -1) 
                self.sigmaK[1,0,2] = tmp_x1x3 / (split[1] - split[0] -1)
                self.sigmaK[1,1,0] = self.sigmaK[1,0,1]
                self.sigmaK[1,1,1] = tmp_x2x2 / (split[1] - split[0]-1)
                self.sigmaK[1,1,2] = tmp_x2x3 / (split[1] - split[0] -1)
                self.sigmaK[1,2,0] = self.sigmaK[1,0,2]
                self.sigmaK[1,2,1] = self.sigmaK[1,1,2] 
                self.sigmaK[1,2,2] = tmp_x3x3 / (split[1] - split[0] -1)
                # Next Class
                tmp_x1x1,tmp_x1x2,tmp_x1x3 = 0,0,0
                tmp_x2x2,tmp_x2x3 = 0,0
                tmp_x3x3 = 0
                for x in range(split[1],split[2]):
                    if Y[x] == 3:
                        tmp = X[x] - self.mu[2] 
                        tmp_x1x1 += tmp[0]*tmp[0] 
                        tmp_x1x2 += tmp[0]*tmp[1] 
                        tmp_x1x3 += tmp[0]*tmp[2]
                        tmp_x2x2 += tmp[1]*tmp[1]
                        tmp_x2x3 += tmp[1]*tmp[2]
                        tmp_x3x3 += tmp[2]*tmp[2] 
                self.sigmaK[2,0,0] = tmp_x1x1 / (split[2]- split[1]-1) 
                self.sigmaK[2,0,1] = tmp_x1x2 /(split[2] - split[1] -1) 
                self.sigmaK[2,0,2] = tmp_x1x3 / (split[2] - split[1] -1)
                self.sigmaK[2,1,0] = self.sigmaK[2,0,1]
                self.sigmaK[2,1,1] = tmp_x2x2 / (split[2] - split[1]-1)
                self.sigmaK[2,1,2] = tmp_x2x3 / (split[2] - split[1] -1)
                self.sigmaK[2,2,0] = self.sigmaK[2,0,2]
                self.sigmaK[2,2,1] = self.sigmaK[2,1,2] 
                self.sigmaK[2,2,2] = tmp_x3x3 / (split[2] - split[1] -1)
            
                
                #print(self.theta)
                #print("mu",self.mu)
                #print("sigma",self.sigmaK) 
                #raise NotImplementedError



        def segment_image(self, img):
                '''
                        Calculate the segmented image using a classifier
                        eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
                        call other functions in this class if needed
                        
                        Inputs:
                                img - original image
                        Outputs:
                                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
                '''
                # YOUR CODE HERE
                rows = img.shape[0]
                cols = img.shape[1] 
                mask_img = np.zeros((rows,cols))
                 
                for i in range(0,rows):
                    for j in range(0,cols):
                        if np.count_nonzero(img[i,j]) == 0:
                            mask_img[i,j] = 0
                        else:
                            argmax = -999999
                            y = None
                            for k in range(0,3):
                                tmp1 = 1/ ((2*np.pi)**(3/2)*np.linalg.det(self.sigmaK[k])**(1/2))
                                p =  img[i,j] - self.mu[k]
                                p1 = p.reshape(1,3)
                                p2 = np.transpose(p1) 
                                t = np.linalg.inv(self.sigmaK[k])  
                                dot = np.dot( p1,t)
                                dot = np.dot(dot,p2)
                                tmp2 = np.exp((-1/2)* dot)
                                arg = np.log(tmp1*tmp2)
                                arg = np.log(self.theta[k]) + arg
                                if arg > argmax:
                                    argmax = arg
                                    y = k
                            #If class was the max
                            if y == 0: 
                                mask_img[i,j] = 1
                            else:
                                mask_img[i,j] = 0 
                #raise NotImplementedError
                return mask_img

        def get_bounding_box(self, img):
                '''
                        Find the bounding box of the blue barrel
                        call other functions in this class if needed
                        
                        Inputs:
                                img - original image
                        Outputs:
                                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                                is from left to right in the image.
                                
                        Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
                '''
                # YOUR CODE HERE
                pass 
                #raise NotImplementedError
                return boxes


if __name__ == '__main__':
    folder = "test"
    #folder = "trainset"
    my_detector = BarrelDetector()
    for filename in os.listdir(folder):
        # read one test image
        if filename[0] == '.': 
            continue 
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100,40,40])
        upper_blue = np.array([140,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(img, lower_blue, upper_blue)
        res = cv2.bitwise_and(img,img,mask = mask)
        
        #Display results:
        #(1) Segmented images
        print("inMask") 
        mask_img = my_detector.segment_image(res)
        print("MaskImage is created")
        
        #(2) Barrel bounding box
        #boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope

