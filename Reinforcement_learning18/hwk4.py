#! /usr/bin/env python
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import random

import os
import time
import gym
from decimal import *

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicLSTMCell
from network import Network


VISUALIZE = True
SEED = 0
MAX_PATH_LENGTH = 500
NUM_EPISODES = 12000
env_name = 'HalfCheetah-v1' 
if VISUALIZE:
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    env = gym.wrappers.Monitor(env, logdir, force=True, video_callable=lambda episode_id: episode_id%logging_interval==0)
env._max_episode_steps = MAX_PATH_LENGTH

# set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# make variable types for automatic setting to GPU or CPU, depending on GPU availability
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("yes")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



if __name__ == "__main__":
    pass
