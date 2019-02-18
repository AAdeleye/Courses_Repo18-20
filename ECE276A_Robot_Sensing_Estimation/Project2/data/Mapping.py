#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:13:19 2019

@author: aadeleye
"""
import numpy as np 
import matplotlib.pyplot as plt
from collections import OrderedDict
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
        Myclock.append( (imu_stamps[i],'i', i))
    for i in range(0,len(encoder_stamps)):
        Myclock.append((encoder_stamps[i],'e', i))
    for i in range(0,len(lidar_stamps)):
        Myclock.append((lidar_stamps[i],'l', i))
    Myclock = sorted(Myclock, key=lambda tup: tup[0])


def Log_odds_update(lidar_scan):
    global robot_pos, MAP
    #Radians 
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    indValid = np.logical_and((lidar_scan < 30),(lidar_scan > 0.2))

    lidar_scan = lidar_scan[indValid]
    angles = angles[indValid]
    
    #Transform Lidar to RobotBody in xy meters
    xs_t = lidar_scan*np.cos(angles) - .13673
    ys_t = lidar_scan*np.sin(angles)
  
    #Tranfrom Lidar from Robot Body to World Frame

    xs_t = robot_pos[0] + xs_t 
    ys_t = robot_pos[0] + ys_t
    
    
    sx = robot_pos[0]
    sy = robot_pos[1]
  
    #Set all squares not at the start or end to log
        #End poing of each scan 
    for i in range(0,len(xs_t)):
        ex = xs_t[i]
        ey = ys_t[i]
        map_updates = bresenham2D(sx, sy, ex, ey)
        len_map = map_updates.shape[1]
       
        #Points from start to right before end
        for ii in range(0,len_map-1):
            LOG_MAP[int(map_updates[0,ii]),int(map_updates[1,ii])] += np.log(1/4)
        #End point (Where the object is), unless it is right in front of us
        if len_map != 1:
            LOG_MAP[int(map_updates[0,len_map-1]),int(map_updates[1,int(len_map-1)])] += np.log(4)
        
    

def Mapping():
  
    #Radians 
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    
  
    # init Screen
    Obs = {}
    Obs['res']   = 0.05 #meters
    Obs['xmin']  = -20  #meters
    Obs['ymin']  = -20
    Obs['xmax']  =  20
    Obs['ymax']  =  20 
    Obs['sizex']  = int(np.ceil((Obs['xmax'] - Obs['xmin']) / Obs['res'] + 1)) #cells
    Obs['sizey']  = int(np.ceil((Obs['ymax'] - Obs['ymin']) / Obs['res'] + 1))
    Obs['map'] = np.zeros((Obs['sizex'],Obs['sizey']),dtype=np.int8) #DATA TYPE: char or int8
    
    plt.ion
    
   
    #imu avg [sum,cnt]
    imu_avg = [0,0]
    #Encoder (value,time_stamp)
    past_encoder = (0,0)
    #Loop through all sensor readings
    for sensor_scan in Myclock:
        if sensor_scan[1] == 'l':
            Log_odds_update(lidar_ranges[:,sensor_scan[2]])
    
        if sensor_scan[1] == 'e':
            #''''We acount for when we start and don't have imu but not for when we end and don't have imu'''
            if imu_avg[0] == 0:
                past_encoder = (encoder_counts[:,sensor_scan[2]],sensor_scan[0])
            else:
                #Predition step
                tv = np.pi * .254 * np.sum(encoder_counts[:,sensor_scan[2]]) 
                tv = tv / 360
                w_i = imu_avg[0] / imu_avg[1] 
                T = sensor_scan[0] - past_encoder[1] 
                a = (w_i * T) /2 
                b = robot_pos[2] + a 
                robot_pos[0] = robot_pos[0] + np.sinc(a) * np.cos(b)
                robot_pos[1] = robot_pos[1] + np.sinc(a) * np.sin(b)
                robot_pos[2] = w_i
                MAP['map'][int(robot_pos[0]),int(robot_pos[1])] = 2
                imu_avg = [0,0]
            
        if sensor_scan[1] == 'i':
            imu_avg[0] += imu_angular_velocity[2,sensor_scan[2]]
            imu_avg[1] += 1 
            pass 
        #Want to Transfrom Lider scan from body to Robot and from 
        # Robot to world 
    
        # build an arbitrary map 
        #indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
        #MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1
        
        #plot lidar points
        #plt.plot(xs_t,ys_t,'.k')
        plt.plot(MAP['map'])
        
        plt.pause(0.001)
        #plt.clf()
    plt.show(block=True)
        
    
    
    
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
    MAP['res']   = 0.1 #meters
    MAP['xmin']  = -500  #meters
    MAP['ymin']  = -500
    MAP['xmax']  =  500
    MAP['ymax']  =  500
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8  
    
    #Log Odds Map
    LOG_MAP = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8  
    
    #Varaibles needed 
    Myclock  = [ ] 
    robot_pos = [501,501,0]

    
    Data_Org()
    Mapping()
    pass 
