"""
Create path planner with Q-learning.
Precise description environment function, can be found in DQN_PATH.py
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random
from random import uniform
import math
import pickle
from matplotlib import style
import time

style.use("ggplot")

MOVE_PENALTY = 1
PEOPLE_PENALTY = 300
TARGET_REWARD = 40
epsilon = 2
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 3000  # how often to play through env visually.

start_q_table = None  # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

AREA_WIDTH = 2
PEOPLES = 2
MAX_FRAMES = 50
RADIUS = 1
START_X = -1
START_Y = -1
EPISODE = 0
peoples_x = []
peoples_y = []
peoples = []
points_rx = []
points_ry = []
points_tx = []
points_ty = []


class robot():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.tx = 0
        self.ty = 0
        self.peoples_array = []
        self.map = []
        self.ep = 0
        self.eps = epsilon

    def sub(self):
        observe = [(int(self.x - self.tx), int(self.y - self.ty))]
        for i in range(PEOPLES):
            observe1 = (int(self.x - self.peoples_array[i].x), int(self.y - self.peoples_array[i].y))
            observe.insert(i + 1, observe1)
        return tuple(observe)

    def get_action_model(self, id):
        action = [[1, 0],
                  [0, 1],
                  [-1, 0],
                  [0, -1],
                  [-1, -1],
                  [-1, 1],
                  [1, -1],
                  [1, 1]]
        return action[id], id

    def calc_reward(self):
        d = []
        for i in range(PEOPLES):
            d.append(math.sqrt(pow(self.peoples_array[i].x - self.x, 2) + pow(self.peoples_array[i].y - self.y, 2)))
        D = min(d)

        if (D < RADIUS / 2):
            reward = -PEOPLE_PENALTY
        elif (math.sqrt(pow(self.tx - self.x, 2) + pow(self.ty - self.y, 2)) < RADIUS / 2):
            reward = TARGET_REWARD
        else:
            reward = -MOVE_PENALTY
        self.distance = math.sqrt(pow(self.tx - self.x, 2) + pow(self.ty - self.y, 2))
        return reward

    def get_state(self):
        pmap = [[0.0 for i in range(int(2 * AREA_WIDTH))] for i in range(int(2 * AREA_WIDTH))]
        self.map = pmap
        self.map[int((AREA_WIDTH + self.x) * 31 / 32)][int((AREA_WIDTH + self.y) * 31 / 32)] = 0.5
        self.map[int((AREA_WIDTH + self.tx) * 31 / 32)][int((AREA_WIDTH + self.ty) * 31 / 32)] = 1
        for i in range(PEOPLES):
            self.map[int((AREA_WIDTH + self.peoples_array[i].x) * 31 / 32)][
                int((AREA_WIDTH + self.peoples_array[i].y) * 31 / 32)] = -1
        return self.map

    def reset(self):
        self.x = START_X
        self.y = START_Y
        self.tx = random.uniform(-AREA_WIDTH, AREA_WIDTH)
        self.ty = random.uniform(-AREA_WIDTH, AREA_WIDTH)
        self.ep = 0
        peoples = []
        for _ in range(PEOPLES):
            peoples.append(people())
        self.peoples_array = peoples
        state = self.get_state()
        return state

    def step(self, action):
        self.x = self.x + action[0] * 0.2
        self.y = self.y + action[1] * 0.2
        if (self.x > AREA_WIDTH):
            self.x = AREA_WIDTH
        if (self.x < -AREA_WIDTH):
            self.x = -AREA_WIDTH
        if (self.y > AREA_WIDTH):
            self.y = AREA_WIDTH
        if (self.y < -AREA_WIDTH):
            self.y = -AREA_WIDTH

        for i in range(PEOPLES):
            self.peoples_array[i].calculate_position()
        state = self.get_state()
        reward = self.calc_reward()
        if ((reward == -PEOPLE_PENALTY) or (reward == TARGET_REWARD)):
            terminated = True
        else:
            terminated = False

        return state, reward, terminated


class people():
    def __init__(self):
        self.x = random.uniform(-AREA_WIDTH, AREA_WIDTH)
        self.y = random.uniform(-AREA_WIDTH, AREA_WIDTH)
        self.vx = random.uniform(-0.2, 0.2)
        self.vy = random.uniform(-0.2, 0.2)

    def calculate_position(self):
        x = self.x + self.vx
        y = self.y + self.vy
        if (x > AREA_WIDTH):
            x = AREA_WIDTH
            self.vx = random.uniform(-0.2, -0.0001)
            self.vy = random.uniform(-0.2, 0.2)
        if (x < -AREA_WIDTH):
            x = -AREA_WIDTH
            self.vx = random.uniform(0.0001, 0.2)
            self.vy = random.uniform(-0.2, 0.2)
        if (y > AREA_WIDTH):
            y = AREA_WIDTH
            self.vx = random.uniform(-0.2, 0.2)
            self.vy = random.uniform(-0.2, -0.0001)
        if (y < -AREA_WIDTH):
            y = -AREA_WIDTH
            self.vx = random.uniform(-0.2, 0.2)
            self.vy = random.uniform(0.0001, 0.2)
        self.x = x
        self.y = y


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-2 * AREA_WIDTH, 2 * AREA_WIDTH + 1):
        for ii in range(-2 * AREA_WIDTH, 2 * AREA_WIDTH + 1):
            for iii in range(-2 * AREA_WIDTH, 2 * AREA_WIDTH + 1):
                for iiii in range(-2 * AREA_WIDTH, 2 * AREA_WIDTH + 1):
                    for iiiii in range(-2 * AREA_WIDTH, 2 * AREA_WIDTH + 1):
                        for iiiiii in range(-2 * AREA_WIDTH, 2 * AREA_WIDTH + 1):
                            q_table[((i, ii), (iii, iiii), (iiiii, iiiiii))] = [np.random.uniform(-10, 0) for i in
                                                                                range(8)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []


def simulation(Simulation):
    info = False
    obs = env.sub()
    # print(obs)
    if (not Simulation):
        if np.random.random() > env.eps:
            # GET THE ACTION
            action, action_id = env.get_action_model(np.argmax(q_table[obs]))
        else:
            action, action_id = env.get_action_model(np.random.randint(0, 8))
        # Take the action!
        _, reward, terminated = env.step(action)

        new_obs = env.sub()
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action_id]

        if terminated == True:
            new_q = reward
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action_id] = new_q

        env.ep += reward
    else:
        action, action_id = env.get_action_model(np.argmax(q_table[obs]))
        _, reward, terminated = env.step(action)
        env.ep += reward
        if (terminated):
            print(env.ep)
    if terminated == True:
        episode_rewards.append(env.ep)
        env.eps *= EPS_DECAY
        info = True
        env.reset()

    return info


def update_data(frame_number):
    # action = env.get_action_model(random.randint(0,7))
    # env.step(action)
    simulation(True)
    for i in range(PEOPLES):
        peoples_x[i] = env.peoples_array[i].x
        peoples_y[i] = env.peoples_array[i].y

    obj1.set_data(env.x, env.y)
    obj2.set_data(env.tx, env.ty)
    # new data to plot
    obj.set_data(peoples_x, peoples_y)
    return None


def anim():
    for i in range(PEOPLES):
        peoples_x.append(env.peoples_array[i].x)
        peoples_y.append(env.peoples_array[i].y)

    points_rx.append(env.x)
    points_ry.append(env.y)
    points_tx.append(env.tx)
    points_ty.append(env.ty)
    # set view limits
    ax = plt.axes()
    ax.set_ylim(-AREA_WIDTH, AREA_WIDTH)
    ax.set_xlim(-AREA_WIDTH, AREA_WIDTH)

    # animation controller - have to be assigned to variable
    anim = animation.FuncAnimation(fig, update_data, MAX_FRAMES, interval=50)  # , repeat=False)
    plt.show()


env = robot()
env.reset()
for i in range(1000000):
    for ii in range(20):
        while (1):
            if (simulation(False)):
                break
    print(i)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.plot(episode_rewards)
plt.ylabel(f"Reward ma")
plt.xlabel("episode #")
plt.show()

env.reset()
fig = plt.figure()
obj, = plt.plot([], [], 'ro')
obj1, = plt.plot([], [], 'yo')
obj2, = plt.plot([], [], 'bo')
anim()
