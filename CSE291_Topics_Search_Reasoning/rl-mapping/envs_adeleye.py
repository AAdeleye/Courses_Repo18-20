import numpy as np
import random
from bresenham import bresenham

def safe_log(x):
    if x <= 0.:
        return 0.
    return np.log(x)
safe_log = np.vectorize(safe_log)

ACTIONS = {
    0: (1, 0, 0), # down
    1: (-1, 0, 0), # up
    2: (0, 1, 0), # right
    3: (0, -1, 0), # left
    4: (0, 0, -30), # rotate left
    5: (0, 0, 30), # rotate right
}

class Pose:
    def __init__(self, x1=0, y1=0, x2=1, y2=1, orientation1=0, orientation2=0):
        self.x1 = x_1
        self.y1 = y_1
        self.orientation1 = orientation1
        
        self.x2 = x2
        self.y2 = y_2
        self.orientation2 = orientation2
    
    def agents(self)
        return  {0:[self.x1,self.y1,orientation1] , 1:[self.x2,self.x2,orientation2]}  

class LocalISM(object):
    def __init__(self, map, span=1, p_correct=.8):
        self.map = map
        self.N = self.map.shape[0]
        self.span = span
        self.p_correct = p_correct

    def log_odds(self, pose):
        l = np.zeros((self.N, self.N))
        
        agents_location = pose.agents 
        for agent in range(0,2):
            x_pose , y_pose  = agents_location[agent][0], agents_location[agent][1]
            x_low, x_high = max(x_pose-self.span, 0), min(x_pose+self.span, self.N-1)
            y_low, y_high = max(y_pose-self.span, 0), min(y_pose+self.span, self.N-1)
            for i in range(x_low, x_high+1):
                for j in range(y_low, y_high+1):
                    if random.random() < self.p_correct:
                        if self.map[i, j] == 0:
                            l[i, j] = np.log((1-self.p_correct) / self.p_correct)
                        else:
                            l[i, j] = np.log(self.p_correct / (1-self.p_correct))
                    else:
                        if self.map[i, j] == 1:
                            l[i, j] = np.log((1-self.p_correct) / self.p_correct)
                        else:
                            l[i, j] = np.log(self.p_correct / (1-self.p_correct))
            l[x_pose, y_pose] = -float("inf")

        return l

class RangeISM(object):
    def __init__(self, map):
        self.map = map
        self.N = self.map.shape[0]

    def log_odds(self, pose):
        print (pose.x, pose.y, pose.orientation)
        l = np.zeros((self.N, self.N))

        b = list(bresenham(pose.x, pose.y, pose.x + 10*int(self.N*np.cos(pose.orientation*np.pi/180)), pose.y + 10*int(self.N*np.sin(pose.orientation*np.pi/180))))
        for i, pos in enumerate(b):
            if b[i+1][0] < 0 or b[i+1][1] < 0 or b[i+1][0] >= self.N or b[i+1][1] >= self.N:
                break
            elif self.map[pos[0], pos[1]]:
                l[pos[0], pos[1]] = 2
                break
            else:
                l[pos[0], pos[1]] = -2
        l[pose.x, pose.y] = -2
        return l

class MappingEnvironment(object):
    def __init__(self, ism_proto, N=10, p=.1, episode_length=1000, prims=False, randompose=True):
        self.ism_proto = ism_proto
        self.N = N
        self.p = p
        self.episode_length = episode_length
        self.prims = prims
        self.random_pose = randompose
        self.t = None
        self.viewer = None

    def reset(self):
        # generate new map
        if self.prims:
            self.map = self.generate_map_prims()
        else:
            self.map = np.random.choice([0, 1], p=[1-self.p, self.p], size=(self.N, self.N))

        # generate initial pose
        if self.random_pose:
            self.a1_x, self.a1_y = np.random.randint(0, self.N), np.random.randint(0, self.N)
            self.a2_x, self.a2_y = np.random.randint(0, self.N), np.random.randint(0, self.N)
            if self.a1_x == self.a2_x and self.a1_y == self.a2_y:
                self.a1_x, self.a1_y = np.random.randint(0, self.N), \ 
                                       np.random.randint(0, self.N)
            
        else:
            self.a1_x, self.a1_y, self.a2_x, self.a2_y = 0, 0, 0, 0
        self.pose = Pose(self.a1_x, self.a1_y,self.a2_x, self.a2_y)
        self.map[self.a1_x, self.a1_y] , self.map[self.a2_x, self.a2_y] = 0, 0

        # reset inverse sensor model, likelihood and pose
        self.ism = self.ism_proto(self.map)
        self.l_t = np.zeros((self.N, self.N))
        self.t = 0

        return self.get_observation()

    def neighbors(self, r, c):
        n = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        nfinal = []
        for dr, dc in n:
            if self.in_map(r+dr, c+dc):
                nfinal.append((r+dr, c+dc))
        return nfinal

    def generate_map_prims(self):
        # Start with a grid of filled cells
        map = np.ones((self.N, self.N))
        visited = np.zeros((self.N, self.N))

        # Pick a cell, mark it as part of the maze
        map[0, 0] = 0
        visited[0, 0] = 1

        # Add the surrounding filled cells of the cell to the cell list
        cell_list = set()
        cell_list.add((0, 1))
        cell_list.add((1, 0))

        # While there are cells in the list
        while len(cell_list) != 0:
            # Pick a random cell from the list and remove it
            sample = random.sample(cell_list, 1)[0]
            x, y = sample
            visited[x, y] = 1
            cell_list.remove(sample)

            # Count the number of explored neighbours
            neighbors = self.neighbors(x, y)
            num_explored = 0
            for n in neighbors:
                if map[n[0], n[1]] == 0.:
                    num_explored += 1

            # If the cell doesn't have 2 explored neighbours
            if num_explored != 2:
                # Clear the cell
                map[x, y] = 0

                # Add the neighbouring filled cells to the cell list
                for n in neighbors:
                    if map[n[0], n[1]] == 1. and visited[n[0], n[1]] == 0.:
                        cell_list.add((n[0], n[1]))
        return map

    def in_map(self, x, y):
        return x >= 0 and y >= 0 and x < self.N and y < self.N

    def legal_change_in_pose(self, pose, dx, dy):
        return self.in_map(pose.x + dx, pose.y + dy) and self.map[pose.x + dx, pose.y + dy] == 0

    def logodds_to_prob(self, l_t):
        return 1 - 1./(1 + np.exp(l_t))

    def calc_entropy(self, l_t):
        p_t = self.logodds_to_prob(l_t)
        entropy = - (p_t * safe_log(p_t) + (1-p_t) * safe_log(1-p_t))

        return entropy

    def observation_size(self):
        return 2*self.N - 1

    def get_observation(self):
        #I think he is centering the belife map on the current pose
        agents_loc = pose.agents
        p = [None, None]
        ent = [None, None]
        for agent in range(0,2):
            x_pose, y_pose = agents_loc[agent][[0], agent_loc[agent][1]
            augmented_p = float("inf")*np.ones((3*self.N-2, 3*self.N-2))
            augmented_p[self.N-1:2*self.N-1, self.N-1:2*self.N-1] = self.l_t
            obs = augmented_p[x_pose:x_pose+2*self.N-1, y_pose:y_pose+2*self.N-1]

            p[agent] = self.logodds_to_prob(obs)
        
            ent = self.calc_entropy(obs)

        # # scale p to [-1, 1]
        p[0] = (p[0] - .5)*2
        p[1] = (p[1] - .5)*2

        # # scale entropy to [-1, 1]
        ent[0] /= -np.log(.5)
        ent[0] = (ent[0] - .5)*2
        ent[1] /= -np.log(.5)
        ent[1] = (ent[1] - .5)*2

        return np.concatenate([np.expand_dims(p[0], -1), np.expand_dims(ent[0], -1), \
                 np.expand_dims(p[1], -1), np.expand_dims(ent[1], -1)], axis=-1)

    def num_channels(self):
        return 4

    def num_actions(self)
        return len(ACTIONS.keys())

    def step(self, a):
        # Step time
        if self.t is None:
            print ("Must call env.reset() before calling step()")
            return
        self.t += 1

        # Perform action
        dx1, dy1, dr1,dx2,dy2,dr2 = ACTIONS[a]
        if self.legal_change_in_pose(self.pose, dx1, dy1, dx2, dy2):
            self.pose.x1 += dx1
            self.pose.y1 += dy1
            self.pose.orientation1 = (self.pose.orientation + dr1) % 360
            self.pose.x2 += dx2
            self.pose.y2 += dy2
            self.pose.orientation2 = (self.pose.orientation + dr2) % 360

        # bayes filter
        new_l_t = self.l_t + self.ism.log_odds(self.pose)

        # reward is decrease in entropy
        reward = np.sum(self.calc_entropy(self.l_t)) - np.sum(self.calc_entropy(new_l_t))

        # Check if done
        done = False
        if self.t == self.episode_length:
            done = True
            self.t = None

        self.l_t = new_l_t

        return self.get_observation(), reward, done, None

    def render(self, reset=False):
        from gym.envs.classic_control import rendering

        if reset:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

        if self.viewer is None:
            self.viewer = rendering.Viewer(1000,500)
            self.viewer.set_bounds(0,2*self.N,0,self.N)

            self.geom_grid = []
            for i in range(self.N):
                geoms = []
                for j in range(self.N):
                    geoms.append(make_box(i+self.N, j, 1, 1))
                    self.viewer.add_geom(geoms[-1])
                    if self.map[i, j] == 1:
                        poly = make_box(i, j, 1, 1, color=(.5, .5, .5))
                        self.viewer.add_geom(poly)
                self.geom_grid.append(geoms)

            self.pos = make_box(self.pose.x1 self.pose.y1, 1, 1, color=(1., 105./255, 180./255))
            self.postrans = rendering.Transform()
            self.postrans.set_translation(self.a1_x, self.a1_y)
            self.pos.add_attr(self.postrans)
            self.viewer.add_geom(self.pos)

            self.pos1 =make_box(self.pose.x1, self.pose.y1, 1, 1, color=(1., 105./255, 180./255))
            self.postrans1 = rendering.Transform()
            self.postrans1.set_translation(self.a1_x+self.N, self.a1_y)
            self.pos1.add_attr(self.postrans1)
            self.viewer.add_geom(self.pos1)
            
            self.pos2 = make_box(self.pose.x2, self.pose.y2, 1, 1, color=(1., 105./255, 180./255))
            self.postrans2 = rendering.Transform()
            self.postrans2.set_translation(self.a2_x, self.a2_y)
            self.pos2.add_attr(self.postrans2)
            self.viewer.add_geom(self.pos2)

            self.pos3 =make_box(self.pose.x2, self.pose.y2, 1, 1, color=(1., 105./255, 180./255))
            self.postrans3 = rendering.Transform()
            self.postrans3.set_translation(self.a2_x+self.N, self.a2_y)
            self.pos3.add_attr(self.postrans3)
            self.viewer.add_geom(self.pos3)

        p = self.logodds_to_prob(self.l_t)
        for i in range(self.N):
            for j in range(self.N):
                self.geom_grid[i][j].set_color(0, p[i, j], 0)

        self.postrans.set_translation(self.pose.x1-self.a1_x, self.pose.y1-self.a1_y)
        self.postrans1.set_translation(self.pose.x1-self.a1_x+self.N, self.pose.y2-self.a1_y)
        self.postrans2.set_translation(self.pose.x2-self.a2_x, self.pose.y2-self.a2_y)
        self.postrans3.set_translation(self.pose.x2-self.x0+self.N, self.pose.y-self.a2_y)

        return self.viewer.render(return_rgb_array = True)

def make_box(x, y, w, h, color=None):
    from gym.envs.classic_control import rendering

    poly = rendering.make_polygon([(x, y), (x, y+h), (x+w, y+h), (x+w, y)], filled=True)
    if color is not None:
        poly.set_color(*color)

    return poly
