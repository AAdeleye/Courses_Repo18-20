import numpy as np
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import  linalg 


def hat_map(x):
    temp = np.zeros((3,3))
    temp[1,0] = x[2]
    temp[2,0] = -x[1]
    temp[0,1] = -x[2]
    temp[2,1] = x[0]
    temp[0,2] = x[1]
    temp[1,2] = -x[0]
    return temp

def pi(x):
    temp = np.zeros((4,1))
    temp[:] = x * (1/x[2])
    return temp

def d_pi(x):
    temp = np.identity(4)
    temp_2 = -x * (1/x[2])
    temp_2 = np.reshape(temp_2,(1,4))
    temp[:,2] = temp_2
    temp[2,2] = 0
    return temp
    
def Mapping():
    global Mu, x_mov, y_mov, z_mov, features 
    ti = T[:,0]
    for i in range(1,T.shape[1]):
        t = T[:,i]
        t = t - ti
        wt_hat = hat_map(rotational_velocity[:,i])
        
        ut_hat = np.zeros((4,4))
        ut_hat[0:3,0:3] = wt_hat
        
        ut_hat[0:3,3] = linear_velocity[:,i]
      
        T_t_i =   linalg.expm(-t*(ut_hat))
        Mu = T_t_i @ Mu
         
        Mu_inv = np.linalg.inv(Mu)
        x_mov.append(Mu_inv[0,3])
        y_mov.append(Mu_inv[1,3])


        # (b) Landmark Mapping via EKF Update

        for zi in range(0,features.shape[1]):
            #Each feacture 
            z_t = features[:,zi,i]
            if z_t[0] != -1:
                
                if Visited_list[zi] == 0:
                    Visited_list[zi] = 1
                    z = M[0,0]*b / (z_t[0] - z_t[2])
                    x = (z_t[0] - M[0,2])* z/M[0,0]
                    y = (z_t[1] - M[1,2]) * z/M[1,1]
                    temp = [x,y,z,1]
                    temp = np.linalg.inv(Mu) @ np.linalg.inv(cam_T_imu) @ np.reshape(temp,(4,1))
                    Mu_j[:,zi] = np.reshape(temp,(1,4))
                    #print("pre M", Mu_j[:,zi])
                
                Mu_j_t = np.reshape(Mu_j[:,zi],(4,1))
                inner_product = cam_T_imu @ Mu @ Mu_j_t
                z_hat = M @ pi(inner_product)
                
                
                H_j = M @ d_pi(inner_product) @ cam_T_imu @ Mu @ D 
                V = np.identity(4) * np.random.randint(100,500)
                K_t =  Cov[zi,:,:] @ np.transpose(H_j) @ np.linalg.inv(H_j @ Cov[zi,:,:] @ np.transpose(H_j)  + V)
                
                
                z_t = np.reshape(z_t,(4,1))  
          
                temp = D @ K_t @ (z_t - z_hat) 
                temp = np.reshape(temp,(1,4))
                
                Mu_j[:,zi] = Mu_j[:,zi] + temp
                
                Cov[zi,:,:] = (np.identity(3) - (K_t @ H_j)) @ Cov[zi,:,:]
                #print("z_hat", z_hat)    
                
        
 
        #Set Past time stamp
        ti = T[:,i]
        

if __name__ == '__main__':
    filename = "./data/0027.npz"
    #filename = "./data/0042.npz"
    T,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    
    # Variables needed 
        #-------------
    x_mov = [ ]
    y_mov = [ ] 
    
    z_landmarks_x = []
    z_landmarks_y = []
    
    #UT = np.concatenate((linear_velocity, rotational_velocity))
    Mu = np.zeros((4,4))
    Mu[3,3] = 1
    Mu[0,0] = 1
    Mu[1,1] = 1
    Mu[2,2] = 1 
    x_mov.append(Mu[0,3])
    y_mov.append(Mu[1,3])
    
    M = np.zeros((4,4))
    M[0:2,0:3] = K[0:2,0:3]
    M[2:4,0:3] = K[0:2,0:3]
    M[2,3] = -K[0,0]*b
    
    Cov = np.zeros((features.shape[1],3,3))
    Cov[:,:,:] = np.identity(3) * 1/1000
    
    Mu_j = np.zeros((4,features.shape[1]))
    
    
    D = np.zeros((4,3))
    D[2,:] = 0
    D[0,0] = 1
    D[1,1] = 1
    D[2,2] = 1 
    
    Visited_list = np.zeros(features.shape[1])
    
    V = np.identity(4) * 100
    #------------------    
    
	# (a) IMU Localization via EKF Prediction
    	# (b) Landmark Mapping via EKF Update
    Mapping()
    
    for x in range(0,features.shape[1]):
        z_landmarks_x.append( Mu_j[:,x][0])
        z_landmarks_y.append( Mu_j[:,x][1])

    

	# (c) Visual-Inertial SLAM (Extra Credit)

	#Visualize Map
        # You can use the function below to visualize the robot pose over time
    fig = plt.figure()
        #ax1 = fig.add_subplot(111,projection='3d') 
    plt.scatter(x_mov,y_mov,1)
    plt.scatter(z_landmarks_x,z_landmarks_y,1)
    	#visualize_trajectory_2d(world_T_imu,show_ori=True)
