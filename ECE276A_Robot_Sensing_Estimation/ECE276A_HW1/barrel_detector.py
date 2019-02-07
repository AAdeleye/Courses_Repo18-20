'''
ECE276A WI19 HW1
Blue Barrel Detector
Akanimoh Adeleye
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np  



class BarrelDetector():
        #Weights 
        ''' The catogory of weights are Blue,NoneBlue,Grey'''
        theat, mu, sigmaK = None, None, None 

        def __init__(self):
                '''
                        Initilize your blue barrel detector with the attributes you need
                        eg. parameters of your classifier
                '''
                self.theta = np.load('theta.npy')
                self.mu = np.load('mu.npy')
                self.sigmaK = np.load('sigma.npy')
                
                print("Theta:\n ", self.theta)
                print("Mu:\n", self.mu)
                print("Sigma:\n", self.sigmaK)

                
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

                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                lower_blue = np.array([100,40,40])
                upper_blue = np.array([140,255,255])

                # Threshold the HSV image to get only blue colors
                mask = cv2.inRange(img, lower_blue, upper_blue)
                img = cv2.bitwise_and(img,img,mask = mask)
                
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
                mask = self.segment_image(img)
                boxes = [] 
                #detector = cv2.SimpleBlobDetector()
                #blobs = detector.detect(mask)
                #blobs = cv2.drawKeypoints(img, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                #cv2.imshow('image', blobs)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
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
        #Display results:
        #(1) Segmented images
        mask_img = my_detector.segment_image(img)

        #(2) Barrel bounding box
        #boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope

