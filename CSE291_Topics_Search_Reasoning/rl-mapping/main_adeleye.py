import argparse
import os
import random

from envs_adeleye import MappingEnvironment, LocalISM, RangeISM
from model import CNNActorCritic, MLPActorCritic, ResNetActorCritic, LinearActorCritic, PolNet
from distributions import Multinomial

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()

# General Stuff
parser.add_argument('--experiment', default='run0/', help='folder to put results of experiment in')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

# Neural Network
parser.add_argument('--network', default='polnet', help='network type: polnet | mlp | cnn | resnet')

# Environment
parser.add_argument('--N', type=int, default=25, help='size of grid')
parser.add_argument('--map_p', type=float, default=.1, help='probability map location is occupied')
parser.add_argument('--prims', action='store_true', help='prims algorithm for filling in map')

# Sensor
parser.add_argument('--sensor_type', default='local', help='local | range')
parser.add_argument('--sensor_span', type=int, default=1, help='span of sensor')
parser.add_argument('--sensor_p', type=float, default=.8, help='probability sensor reading is correct')

# MDP
parser.add_argument('--gamma', type=float, default=.99, help='discount rate')
parser.add_argument('--episode_length', type=int, default=300, help='length of mapping environment episodes')

# Training
parser.add_argument('--N_episodes', type=int, default=20000, help='number of episodes to train for')
parser.add_argument('--max_steps', type=int, default=20, help='number of forward steps in A2C')
parser.add_argument('--optimizer', default='adam', help='sgd | adam | rmsprop')
parser.add_argument('--anneal_step_size', type=int, default=5000, help='number of episodes until anneal learning rate')
parser.add_argument('--anneal_gamma', type=float, default=.5, help='annealing multiplicative factor')
parser.add_argument('--lr', type=float, default=10e-8, help='learning rate for ADAM optimizer')
parser.add_argument('--lambda_entropy', type=float, default=.001, help='entropy term coefficient')
parser.add_argument('--max_grad_norm', type=float, default=50., help='max gradient norm of actor_critic')
parser.add_argument('--seed', type=int, default=random.randint(0, 10000), help='random seed')

opt = parser.parse_args()
opt.cuda = opt.cuda and torch.cuda.is_available()
print(opt)

# set random seeds
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

# make experiment path
os.makedirs(opt.experiment, exist_ok=True)
with open(os.path.join(opt.experiment, 'config.txt'), 'w') as f:
    f.write(str(opt))

# Initialize sensor
if opt.sensor_type == 'local':
    ism_proto = lambda x: LocalISM(x, span=opt.sensor_span, p_correct=opt.sensor_p)
elif opt.sensor_type == 'range':
    ism_proto = lambda x: RangeISM(x)
else:
    raise Exception('sensor type not supported.')

# Initialize environment
env = MappingEnvironment(ism_proto, N=opt.N, p=opt.map_p, episode_length=opt.episode_length, prims=opt.prims)

# Initialize actor critic neural network
if opt.network == 'cnn':
    actor_critic = CNNActorCritic(H_in = env.observation_size(), nc = env.num_channels(), na = env.num_actions())
elif opt.network == 'mlp':
    actor_critic = MLPActorCritic(H_in = env.observation_size(), nc = env.num_channels(), na = env.num_actions())
elif opt.network == 'resnet':
    actor_critic = ResNetActorCritic(H_in = env.observation_size(), nc = env.num_channels(), na = env.num_actions())
elif opt.network == 'polnet':
    policy_gradient_net = PolNet(H_in = env.observation_size(), nc = env.num_channels(), na = env.num_actions())
else:
    raise Exception('network type not supported')

if opt.cuda:
    #actor_critic = actor_critic.cuda()
    policy_gradient_net = policy_gradient_net.cuda()

# Initialize optimizer and learning rate scheduler
if opt.optimizer == 'rmsprop':
    actor_critic_optimizer = torch.optim.RMSprop(actor_critic.parameters(), lr=opt.lr)
elif opt.optimizer == 'adam':
    #actor_critic_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=opt.lr)
    policy_gradient_optimizer = torch.optim.Adam(policy_gradient_net.parameters(), lr=opt.lr)
elif opt.optimizer == 'sgd':
    actor_critic_optimizer = torch.optim.SGD(actor_critic.parameters(), lr=opt.lr)
else:
    raise Exception('optimizer not supported. Try rmsprop/adam/sgd')

# Initialize necessary variables
obs = env.reset()
#env.render(reset=True)
done = False
t = 0
episodes = 0
ep_rewards = [0]

print ("Main training loop")
while episodes < opt.N_episodes:
    t_start = t
    rewards, observations, actions = [], [], []

    # on-policy training loop for max_steps timesteps
    while 1:
        # Perform a_t according to Policy Gradient
        obs_npy = obs.transpose(2, 0, 1)[None, :]
        obst = torch.Tensor(obs_npy)
        if opt.cuda:
            obst = obst.cuda()
        obsv = Variable(obst)
        policy_gradient_net.eval()
        pa1,pa2 = policy_gradient_net(obsv)
        pa1 = Multinomial(pa1)
        pa2 = Multinomial(pa2)
        a1 = pa1.sample().data[0]
        a2 = pa2.sample().data[0]
        
        # Receive reward r_t and new state s_t+1
        obs, reward, done, info = env.step(a1,a2)
        #env.render()
        t += 1

        observations.append(obs_npy)
        actions.append([a1,a2])
        rewards.append(reward)
        ep_rewards[-1] += reward 

        if done: # terminal s_t
            R = 0
            episodes += 1
            obs = env.reset()
            #env.render(reset=True)

            print ("Finished Episode %d:" % episodes, ep_rewards[-1], np.mean(ep_rewards[-50:]))
            ep_rewards.append(0.)

            if episodes > 0 and episodes % opt.anneal_step_size == 0:
                print ("Annealing learning rate: %.7f to %.7f" % (opt.lr, opt.lr*opt.anneal_gamma))
                opt.lr *= opt.anneal_gamma
                if opt.optimizer == 'rmsprop':
                    actor_critic_optimizer = torch.optim.RMSprop(actor_critic.parameters(), lr=opt.lr)
                elif opt.optimizer == 'adam':
                    pass 
                    #actor_critic_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=opt.lr)
                    # policy_gradient_optimizer = torch.optim.Adam(policy_gradient_net.parameters(), lr=opt.lr)
                elif opt.optimizer == 'sgd':
                    actor_critic_optimizer = torch.optim.SGD(actor_critic.parameters(), lr=opt.lr)
            break

        if t - t_start == opt.max_steps: # reached num forward steps
            #R = V.data[0]
            R= 0
            break

    # accumulate rewards for advantage calculation
    i = len(rewards)-1
    for r in rewards[::-1]:
        R = rewards[i] + opt.gamma*R
        rewards[i] = R
        i -= 1

    actions_t1 = torch.Tensor([item[0] for item in actions]).type(torch.LongTensor)
    actions_t2 = torch.Tensor([item[1] for item in actions]).type(torch.LongTensor)
    if opt.cuda:
        actions_t1 = actions_t1.cuda()
        actions_t2 = actions_t2.cuda()
    actions_v1 = Variable(actions_t1)
    actions_v2 = Variable(actions_t2)

    rewards_t = torch.Tensor(rewards)
    if opt.cuda:
        rewards_t = rewards_t.cuda()
    eps = np.finfo(np.float32).eps.item()
    rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + eps)
    rewards_v = Variable(rewards_t)

    observations_npy = np.concatenate(observations)
    observations_t = torch.Tensor(observations_npy)
    if opt.cuda:
        observations_t = observations_t.cuda()
    observations_v = Variable(observations_t)

    policy_gradient_net.train()
    pa1,pa2 = policy_gradient_net(observations_v)
    pa1_multinomial = Multinomial(pa1)
    pa2_multinomial = Multinomial(pa2)

    #policy_gradient_net.zero_grad()
    policy_gradient_optimizer.zero_grad()

    # gradient step
    policy_loss1 = -pa1_multinomial.log_prob(actions_v1) * (rewards_v)
    policy_loss2 = -pa2_multinomial.log_prob(actions_v2) * (rewards_v)

    #value_loss = (rewards_v - V).pow(2).mean()
    #entropy1 = -torch.sum(pa1 * torch.log(pa1), dim=1).mean()
    #entropy2 = -torch.sum(pa2 * torch.log(pa2), dim=1).mean()
    
    policy_lossT = (policy_loss1) + (policy_loss2)
    print(policy_lossT)
    (policy_lossT).sum().backward()

    #torch.nn.utils.clip_grad_norm(policy_gradient_net.parameters(), opt.max_grad_norm)
    policy_gradient_optimizer.step()

    np.save(os.path.join(opt.experiment, 'results'), ep_rewards)

    if episodes % 1000 == 0:
        torch.save(policy_gradient_net.state_dict(), os.path.join(opt.experiment, 'actor_critic_episode%d.torch' % episodes))

torch.save(policy_gradient_net.state_dict(), os.path.join(opt.experiment, 'actor_critic_episode%d.torch' % episodes))
np.save(os.path.join(opt.experiment, 'results'), ep_rewards)

rewards = []
for k in range(1000):
    obs = env.reset()
    #env.render(reset=True)

    done = False
    R = 0
    while not done:
        # Perform a_t according to actor_critic
        obs_npy = obs.transpose(2, 0, 1)[None, :]
        obst = torch.Tensor(obs_npy)
        if opt.cuda:
            obst = obst.cuda()
        obsv = Variable(obst)
        policy_gradient_net.eval()
        pa1,pa2 = policy_gradient_net(obsv)
        pa1 = Multinomial(pa1)
        pa2 = Multinomial(pa2)
        a1 = pa1.sample().data[0]
        a2 = pa2.sample().data[0]

        # Receive reward r_t and new state s_t+1
        obs, reward, done, info = env.step(a1,a2)
        #env.render()

        R += reward
    print (R)
    rewards.append(R)

np.save(os.path.join(opt.experiment, 'rewards_test'), rewards)
