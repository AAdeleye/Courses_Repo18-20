import sys
sys.path.append('/datasets/home/22/422/ee276cao/.mujoco/mjpro150/bin')

import gym

print  ("Making CartPole")
gym.make('CartPole-v0')

print ("Making Humanoid")
gym.make('Humanoid-v2')

print ("Making SpaceInvaders")
gym.make('SpaceInvaders-v0')

print ("Making Robotics")
gym.make('HandManipulateBlock-v0')
