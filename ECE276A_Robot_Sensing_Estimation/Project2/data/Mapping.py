#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:13:19 2019

@author: aadeleye
"""
import numpy as np 
import matplotlib.pyplot as plt
#from collections import OrderedDict
from scipy import signal

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))
    
def Data_Org():
    '''
    This function orginzes the data into a dictionary so it is easaily traverable
    MyClock is set as such: [ (Timestamp, sensor_label, matrix_placement),.. ]
    '''
    global Myclock
    for i in range(0,len(imu_stamps)):
        Myclock.append((imu_stamps[i],'i', i))
        
    for i in range(0,len(encoder_stamps)):
        Myclock.append((encoder_stamps[i],'e', i))
        
    for i in range(0,len(lidar_stamps)):
        Myclock.append((lidar_stamps[i],'l', i))
        
    Myclock = sorted(Myclock, key=lambda tup: tup[0])

def binary_map():
    global LOG_MAP
    temp = 1- (1/(1+np.exp(LOG_MAP)))
    temp[temp >= 0.8 ] = 1
    temp[temp < 0.8 ] = 0 
    return temp

    
def Log_odds_update(lidar_scan):
    global robot_pos, MAP, LOG_MAP
    #Radians 
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    indValid = np.logical_and((lidar_scan < 30),(lidar_scan > 0.2))

    lidar_scan = lidar_scan[indValid]
    angles = angles[indValid]
    
    #Transform Lidar to RobotBody in xy meters
        #Lidar_scan is an array ~1081
    xs_t = lidar_scan*np.cos(angles) - .13673
    ys_t = lidar_scan*np.sin(angles)
    
    #Tranfrom Lidar from Robot Body to World Frame
        #Rotoation
    c, s = np.cos(robot_pos[2]), np.sin(robot_pos[2])
    R = np.array(((c,-s), (s, c)))
    lidar_scan = [xs_t,ys_t]
    xs_t, ys_t = np.dot(R,lidar_scan)
        #Translation
    xs_t = robot_pos[0] + xs_t 
    ys_t = robot_pos[1] + ys_t
    
    
    #Start pos, where the robot is in meters converted to cells
    sx = np.ceil((robot_pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    sy = np.ceil((robot_pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
        
    #Set all squares not at the start or end to log
        #End poing of each scan 
    
  
    ex = np.ceil((xs_t - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    ey = np.ceil((ys_t - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    #plt.ion
    for i in range(0,len(ex)):
        xz, yz = bresenham2D(sx, sy, ex[i], ey[i])
        xz = xz.astype(np.int16)
        yz = yz.astype(np.int16)
  
        #print("x,y", xz,yz)
        LOG_MAP[xz[0:len(xz)-1],yz[0:len(yz)-1]] += np.log(1/4)
        #np.add(a, -100, out=a, casting="unsafe") #np.log(1/4)
        #print(xz[len(xz)-1],yz[len(yz)-1])
        LOG_MAP[ xz[len(xz)-1],yz[len(yz)-1]] +=  np.log(4)
    
        
        #plt.pause(0.001)
        #plt.clf()
    #plt.show(block=True)
        
    '''
    #Points from start to right before end
    for ii in range(0,len_map-1):
        LOG_MAP[int(map_updates[0,ii]),int(map_updates[1,ii])] += np.log(1/4)
    #End point (Where the object is), unless it is right in front of us
    if len_map != 1:
        LOG_MAP[int(map_updates[0,len_map-1]),int(map_updates[1,int(len_map-1)])] += np.log(4)
    '''

def Mapping():
  

    #plt.ion
   
    #imu avg [sum,cnt]
    imu_avg = [0,0]
    imu = 0 
    #count
    cnt = 0
    #Loop through all sensor readings
    for sensor_scan in Myclock:
        #cnt += 1
        #print(cnt)
        
        #Sensor_scan =  [Time_stamp, symblo, array_place]
        if sensor_scan[1] == 'l':
            #pass 
            cnt += 1
            Log_odds_update(lidar_ranges[:,sensor_scan[2]])
            print(cnt)
            #if cnt == 41:
                #break 
        if sensor_scan[1] == 'e':
            #''''We acount for when we start and don't have imu but not for when we end and don't have imu''' 
            if imu == 0:
                pass 
                #past_encoder = (encoder_counts[:,sensor_scan[2]],sensor_scan[0])
            else:
                encoder = encoder_counts[:,sensor_scan[2]]
                
                #Predition step
                tv_l =  (encoder[1] + encoder[3] ) /2
                tv_l = tv_l * .0022
                
                tv_r = (encoder[0] + encoder[2] ) /2 
                tv_r = tv_r *.0022
                
                tv = (tv_l + tv_r) / 2 
                
               
                w_i = imu # imu_avg[0] / imu_avg[1] 
                T = sensor_scan[0] - encoder_stamps[sensor_scan[2]-1]
                
                a = (w_i * T) /2 
                b = robot_pos[2] + a 
                
                robot_pos[0] = robot_pos[0] + (np.sinc(a) * np.cos(b))* tv
                robot_pos[1] = robot_pos[1] + (np.sinc(a) * np.sin(b))* tv
                robot_pos[2] = robot_pos[2] + (w_i * T)
                
                #--------------------
                #Mapping 
                 # convert from meters to cells
                xis = np.ceil((robot_pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
                yis = np.ceil((robot_pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
                #print(xis,yis)
                #print(imu)
                x_mov.append(xis)
                y_mov.append(yis)
             
                past_encoder = sensor_scan[0]
              
                # Computer Corrilationa and find best partical
                    #Set robot pose ot best partical
                
        if sensor_scan[1] == 'i':
            imu = imu_angular_velocity[2,sensor_scan[2]]
            #imu_avg[1] += 1
            #imu_avg[0] += imu_angular_velocity[2,sensor_scan[2]]
             
        
        #Want to Transfrom Lider scan from body to Robot and from 
        # Robot to world 
    
        # build an arbitrary map 
        #indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
        #MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1
        
        #plot lidar points
        #plt.plot(xs_t,ys_t,'.k')
        #plt.plot(MAP['map'])
        #plt.savefig('Plot_pics/t'+ str(cnt))
        
        #plt.pause(0.001)
        #plt.clf()
    #plt.show(block=True)
        
    
    
    
if __name__ == '__main__':
    dataset = 20
  
    with np.load("Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps

    with np.load("Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
        
    with np.load("Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
      
    with np.load("Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
    
    
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -40  #meters
    MAP['ymin']  = -40
    MAP['xmax']  =  40
    MAP['ymax']  =  40
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8  
    
    #Log Odds Map
    LOG_MAP = np.zeros((MAP['sizex'],MAP['sizey'])) #DATA TYPE: char or int8  

    print(LOG_MAP.shape)
    
    #Varaibles needed 
    Myclock  = [ ] 
        #ROBOT MOVEMENTS FOR PLOTING
    x_mov = []
    y_mov = [] 
    
    robo_p = np.zeros((2,5))
    robo_d =np.zeros((2,5))
    robo_w = np.zeros((2,5))
    #robot_pos = [0,0,0]
    
    xis = np.ceil((robot_pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((robot_pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    x_mov.append(xis)
    y_mov.append(yis)
  
    

    Data_Org()
    Mapping()
    
    fig, ax = plt.subplots()
    im = ax.imshow(LOG_MAP)
    
    MAP['map'] = binary_map()
    fig, ax = plt.subplots()
    im = ax.imshow(MAP['map'])
    
    plt.scatter(y_mov,x_mov,1)
    #plt.colorbar()
    plt.show()
    plt.savefig('Plot_pics/t')#+ str(cnt))
    pass 
