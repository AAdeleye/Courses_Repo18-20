#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:13:19 2019

@author: aadeleye
"""
import numpy as np 
import matplotlib.pyplot as plt
#from collections import OrderedDict
#from scipy import signal





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
    global LOG_MAP , VIZ_MAP
    temp = 1- (1/(1+np.exp(LOG_MAP)))
    temp[temp >= 0.8 ] = 1
    temp[temp < 0.8 ] = 0 
    
    VIZ_MAP[temp == 1] = 1

    return temp

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


def weight_update(x,y):
    global current_pos, robot_weight, robot_pos 
    bin_map = binary_map()

    Max_corr = np.zeros(len(robot_weight))
    for i in range(0,len(robot_pos[0,:])):
        pos = robot_pos[:,i]
        #Tranfrom Lidar from Robot Body to World Frame
        #Rotoation
        c, s = np.cos(pos[2]), np.sin(pos[2])
        R = np.array(((c,-s), (s, c)))
        lidar_scan = [x,y]
        xs_t, ys_t = np.dot(R,lidar_scan)
            #Translation
        xs_t = pos[0] + xs_t 
        ys_t = pos[1] + ys_t
        
          
        # convert position in the map frame here
        Y = np.stack((xs_t,ys_t))
        

        x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
        y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
        
        #X and Y in Cell
        pos_x = np.ceil((pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        pos_y = np.ceil((pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
       
        x_range = np.arange(pos_x-4,pos_x+5,1)
        y_range = np.arange(pos_y-4,pos_y+5,1)
        x_range = (x_range +1.0) * MAP['res'] + MAP['xmin']
        y_range = (y_range +1.0) * MAP['res'] + MAP['xmin']
  
        
        c = mapCorrelation(bin_map,x_im,y_im,Y,x_range,y_range)
        max_i = np.max(c)
        max_i = np.argwhere(c==max_i)[0]
        c = c[max_i[0],max_i[1]]
        #print('i,c',i,c)
        Max_corr[i] = c 
   
    max_c = np.max(Max_corr)
    robot_weight[:] = robot_weight[:] * np.exp(Max_corr[:]-max_c)
    #robot_weight[:] = robot_weight[:] / np.sum(robot_weight[:])
    #print('rw after coor', robot_weight)
    best_pos = robot_pos[:,np.argmax(robot_weight)]
    neff = np.sum(robot_weight)
    #print("meff", neff)
    if neff < 1: 
        #print('particals over weight',robot_pos[:,robot_weight > .05])
        #print(best_pos)
        robot_pos[:] = best_pos.reshape(3,1)
        #print(robot_pos)
        robot_weight = np.ones(N) * 1/N 
        #print("robot_w", robot_weight)
    #if robot_weight[robot_weight < .5]
    #print('r_weight', robot_weight)
    return best_pos 

    
def Log_odds_update(lidar_scan):
    global robot_pos, MAP, LOG_MAP, VIZ_MAP, current_pos
    '''
    Handel Lider scan: transfrom to robo frame
    transfrom to world frame based in each partical
    (i.e happens in weight update)
    '''
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    indValid = np.logical_and((lidar_scan < 30),(lidar_scan > 0.2))

    lidar_scan = lidar_scan[indValid]
    angles = angles[indValid]
    
    #Transform Lidar to RobotBody in xy meters
        #Lidar_scan is an array ~1081
    xs_t = lidar_scan*np.cos(angles) - .13673
    ys_t = lidar_scan*np.sin(angles)
    
    '''
    #Rotoation
    c, s = np.cos(robot_pos[:,0][2]), np.sin(robot_pos[:,0][2])
    R = np.array(((c,-s), (s, c)))
    lidar_scan = [xs_t,ys_t]
    xs_t, ys_t = np.dot(R,lidar_scan)
        #Translation
    xs_t = robot_pos[0] + xs_t 
    ys_t = robot_pos[1] + ys_t
        
     '''   

    '''
    Update partical weights based on scan, pick the best partical.
    That is where we actually moved to. 
    '''

    # Computer Corrilationa and find best partical
        #Set robot pose ot best partical
    current_pos = weight_update(xs_t,ys_t)
    #current_pos = robot_pos[:,0]
        #Rotoation
    c, s = np.cos(current_pos[2]), np.sin(current_pos[2])
    R = np.array(((c,-s), (s, c)))
    lidar_scan = [xs_t,ys_t]
    xs_t, ys_t = np.dot(R,lidar_scan)
        #Translation
    xs_t = current_pos[0] + xs_t 
    ys_t = current_pos[1] + ys_t
        
    #--------------------
    #Mapping 
     # convert from meters to cells
    xis = np.ceil((current_pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((current_pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    #print(xis,yis)
    #print(imu)
    x_mov.append(xis)
    y_mov.append(yis)

    
    '''
    Now that we know where we moved, updated the map based on this movement.
    '''
    
    #Start pos, where the robot is in meters converted to cells
    sx = np.ceil((current_pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    sy = np.ceil((current_pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
        
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
        VIZ_MAP[xz[0:len(xz)-1],yz[0:len(yz)-1]] = 0 
        #np.add(a, -100, out=a, casting="unsafe") #np.log(1/4)
        #print(xz[len(xz)-1],yz[len(yz)-1])
        LOG_MAP[ xz[len(xz)-1],yz[len(yz)-1]] +=  np.log(4)
        
    
        
        #plt.pause(0.001)
        #plt.clf()
    #plt.show(block=True)
    return (xis,yis)


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
            x,y = Log_odds_update(lidar_ranges[:,sensor_scan[2]])
            print(cnt)
            
            if prints[prints == cnt]:
                plt.imshow(VIZ_MAP)
                plt.scatter(y,x,1)
                plt.savefig('../../../../Plot_pics/Slam/t_' + str(cnt))
                plt.clf()
                
            #if cnt == 200:
             #   break 
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
                
                T = sensor_scan[0] - encoder_stamps[sensor_scan[2]-1]
                w_i = imu # imu_avg[0] / imu_avg[1] 
                
                b = w_i * T 
                a = tv * np.sinc(b/2)
                c = np.cos(robot_pos[2,:] + b/2)
                d = np.sin(robot_pos[2,:] + b/2) 
                robot_pos[0,:] = robot_pos[0,:] + a*c 
                robot_pos[1,:] = robot_pos[1,:] + a*d
                robot_pos[2,:] = robot_pos[2,:] + b

                
                #Add Gausian Noise to robot motion
                partical_len = len(robot_pos[0,:])
                for i in range(1,partical_len):
                    robot_pos[0,i] = robot_pos[0,i] + np.random.normal(0,.3,1)
                    robot_pos[1,i] = robot_pos[1,i] + np.random.normal(0,.3,1)
                    robot_pos[2,i] = robot_pos[2,i] + np.random.normal(0,.001,1)
                
                
                
                '''
                #--------------------
                #Mapping 
                 # convert from meters to cells
                xis = np.ceil((current_pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
                yis = np.ceil((current_pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
                #print(xis,yis)
                #print(imu)
                x_mov.append(xis)
                y_mov.append(yis)
                '''
                
                past_encoder = sensor_scan[0]
              
                
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
    dataset = 23
    # dataset = 23
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
    ''' 
    with np.load("Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
    '''
    
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    MAP['xmax']  =  35
    MAP['ymax']  =  35
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8  
    
    #Log Odds Map
    LOG_MAP = np.zeros((MAP['sizex'],MAP['sizey'])) #DATA TYPE: char or int8 
    
    VIZ_MAP = np.ones((MAP['sizex'],MAP['sizey']))* 2 #DATA TYPE: char or int8 

    
    #Varaibles needed 
    Myclock  = [ ] 
        #ROBOT MOVEMENTS FOR PLOTING
    x_mov = []
    y_mov = [] 
    
    N = 70
    robot_pos = np.zeros((3,N))
    robot_weight = np.ones(N) * 1/N 
    current_pos = [0,0,0]
    #robot_pos = [0,0,0]
    
    #xis = np.ceil((robot_pos[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    #yis = np.ceil((robot_pos[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    #x_mov.append(xis)
    #y_mov.append(yis)
  
    
    prints = np.arange(300,4962,100)
    #np.append(prints,4962)
    Data_Org()
    Mapping()
    
    #fig, ax = plt.subplots()
    #im = ax.imshow(VIZ_MAP)
    plt.imshow(VIZ_MAP)
    plt.scatter(y_mov,x_mov,1)
    #plt.show()
    
    MAP['map'] = binary_map()
    #fig, ax = plt.subplots()
    #im = ax.imshow(MAP['map'])
    
    #plt.scatter(y_mov,x_mov,1)
    #plt.colorbar()
    #plt.show()
    plt.savefig('../../../../Plot_pics/Slam/t_2')#+ str(cnt))
    pass 
