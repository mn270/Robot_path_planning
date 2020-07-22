"""
Realization DDQN robot path planning in dynamic environment
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random
from random import uniform
import math
# import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import History
import tensorflow as tf
from collections import deque
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard
import pickle
from matplotlib import style
from time import time
import os

# turn off warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

style.use("ggplot")

MOVE_PENALTY = 0.01
PEOPLE_PENALTY = 1
TARGET_REWARD = 1
epsilon = 1
EPSILON_DECAY = 0.99978
MIN_EPSILON = 0.001
SHOW_EVERY = 3000  # how often to play through env visually.
REPLAY_MEMORY_SIZE = 100_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64
# How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

DISCOUNT = 0.85

AREA_WIDTH = 5
PEOPLES = 2
MAX_FRAMES = 50
RADIUS = 1
START_X = -2
START_Y = -2
EPISODE = 0
peoples_x = []
peoples_y = []
peoples = []
points_rx = []
points_ry = []
points_tx = []
points_ty = []

history = History()


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
        self.steps = 0

    def sub(self):
        """
        Create vector of observation, distance between robot, obstacles and target.
        :return: (tuple []) observation
        """
        observe = [(int(self.x - self.tx), int(self.y - self.ty))]
        for i in range(PEOPLES):
            observe1 = (int(self.x - self.peoples_array[i].x), int(self.y - self.peoples_array[i].y))
            observe.insert(i + 1, observe1)
        return tuple(observe)

    def get_action_model(self, id):
        """
        Encode action vector (movement).
        :param id: (int) action_id
        :return: (int []) action, (int) action_id
        """
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
        """
        Caculate reward, collision detect.
        :return: (float) reward
        """
        collision = False
        for i in range(PEOPLES):
            if (self.peoples_array[i].x == self.x and self.peoples_array[i].y == self.y):
                collision = True

        if (collision):
            reward = -PEOPLE_PENALTY
        elif (self.x == int(self.tx) and self.y == int(
                self.ty)):
            reward = TARGET_REWARD
        else:
            reward = -MOVE_PENALTY
        self.distance = math.sqrt(pow(self.tx - self.x, 2) + pow(self.ty - self.y, 2))
        return reward

    def get_state(self):
        """
        Create a map - image with 3 channels
        1st channel - robot
        2sd channel - target
        3rd channel - obstacles
        0 - empty position
        1 - busy position
        :return: (float []) map
        """
        pmap = np.zeros((2 * AREA_WIDTH + 1, 2 * AREA_WIDTH + 1, 3), dtype=float)
        pmap[int(AREA_WIDTH + self.x)][int(AREA_WIDTH + self.y)] += [1, 0, 0]
        pmap[int(AREA_WIDTH + self.tx)][int(AREA_WIDTH + self.ty)] += [0, 1, 0]
        for i in range(PEOPLES):
            pmap[int(AREA_WIDTH + self.peoples_array[i].x)][int(AREA_WIDTH + self.peoples_array[i].y)] += [0, 0, 1]

        self.map = pmap
        return self.map

    def reset(self):
        """
        Reset environment
        :return: (float []) map
        """
        self.x = random.randint(-AREA_WIDTH, AREA_WIDTH)
        self.y = random.randint(-AREA_WIDTH, AREA_WIDTH)
        self.tx = random.randint(-AREA_WIDTH, AREA_WIDTH)
        self.ty = random.randint(-AREA_WIDTH, AREA_WIDTH)
        if (math.sqrt(pow(self.tx - self.x, 2) + pow(self.ty - self.y, 2)) < 2):
            self.tx = -self.x
            self.ty = -self.y
        self.ep = 0
        peoples = []
        for i in range(PEOPLES):
            peoples.append(people())
            if (math.sqrt(pow(peoples[i].x - self.x, 2) + pow(peoples[i].y - self.y, 2)) < 2):
                peoples[i].x = -self.x
                peoples[i].y = -self.y
        self.peoples_array = peoples
        state = self.get_state()
        self.steps = 0
        return state

    def step(self, action):
        """
        1) Change robot position.
        2) Change obstacles position.
        3) Calculate reward.
        4) Check ending condition.
        :param action: (int [])
        :return: (float []) state, (float) reward, (bool) terminated
        """
        self.steps += 1
        self.x = self.x + action[0]
        self.y = self.y + action[1]
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
        if ((reward == -PEOPLE_PENALTY) or (reward == TARGET_REWARD) or (self.steps >= 200)):
            terminated = True
        else:
            terminated = False

        return state, reward, terminated


class people():
    """
    Dynamic obstacles class (people)
    """
    def __init__(self):
        self.x = random.randint(-AREA_WIDTH, AREA_WIDTH)
        self.y = random.randint(-AREA_WIDTH, AREA_WIDTH)
        self.vx = random.randint(-1, 1)  # random.uniform(-0.2,0.2)
        self.vy = random.randint(-1, 1)  # random.uniform(-0.2,0.2)

    def calculate_position(self):
        """
        Calculate obstacles position
        :return: NULL
        """
        x = self.x + self.vx
        y = self.y + self.vy
        if (x > AREA_WIDTH):
            x = AREA_WIDTH
            self.vx = -1  # random.uniform(-0.2,-0.0001)
            self.vy = random.randint(-1, 1)  # random.uniform(-0.2, 0.2)
        if (x < -AREA_WIDTH):
            x = -AREA_WIDTH
            self.vx = 1  # random.uniform(0.0001,0.2)
            self.vy = random.randint(-1, 1)  # random.uniform(-0.2, 0.2)
        if (y > AREA_WIDTH):
            y = AREA_WIDTH
            self.vx = random.randint(-1, 1)  # random.uniform(-0.2, 0.2)
            self.vy = -1  # random.uniform(-0.2,-0.0001)
        if (y < -AREA_WIDTH):
            y = -AREA_WIDTH
            self.vx = random.randint(-1, 1)  # random.uniform(-0.2, 0.2)
            self.vy = 1  # random.uniform(0.0001,0.2)
        self.x = x
        self.y = y


class DQNAgent:
    """
    Create DQN model and save history.
    """
    def __init__(self):

        # Main model
        self.model = self.create_model()
        self.history = None
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    def create_model(self):
        """
        Create model
        :return: model
        """
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(2 * AREA_WIDTH + 1, 2 * AREA_WIDTH + 1, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), input_shape=(2 * AREA_WIDTH + 1, 2 * AREA_WIDTH + 1, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dense(8, activation='linear'))
        model.compile(loss="huber_loss", optimizer='Adam', metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.history = self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                                      callbacks=[self.tensorboard])

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        accurancy.append(self.history.history['accuracy'][0])
        loss.append(self.history.history['loss'][0])

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


episode_rewards = []
agent = DQNAgent()
accurancy = []
loss = []


def simulation(Simulation):
    info = False
    obs = env.sub()
    # print(obs)
    if (not Simulation):
        episode_reward = 0
        step = 1
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > env.eps:
                # Get action from Q table
                id = np.argmax(agent.get_qs(current_state))
                action, _ = env.get_action_model(id)
            else:
                # Get random action
                id = np.random.randint(0, 8)
                action, _ = env.get_action_model(id)

            new_state, reward, done = env.step(action)

            # Transform new continous state to new discrete state and count reward
            env.ep += reward

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, id, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)

        # Decay epsilon
        if env.eps > MIN_EPSILON:
            env.eps *= EPSILON_DECAY
            env.eps = max(MIN_EPSILON, env.eps)
    else:
        action, _ = env.get_action_model(np.argmax(agent.get_qs(env.map)))
        new_state, reward, done = env.step(action)

    if done == True:
        episode_rewards.append(env.ep)
        env.eps *= EPSILON_DECAY
        info = True
        env.reset()

    return info


def update_data(frame_number):
    simulation(True)
    for i in range(PEOPLES):
        peoples_x[i] = env.peoples_array[i].x
        peoples_y[i] = env.peoples_array[i].y

    obj1.set_data(env.x, env.y)
    obj2.set_data(env.tx, env.ty)
    # new data to plot
    obj.set_data(peoples_x, peoples_y)
    return None


def anim(plt):
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
eps_history = []
k = 16
for i in tqdm(range(1000), ascii=True, unit='episodes'):
    for ii in tqdm(range(k), ascii=True, unit='step'):
        while (1):
            if (simulation(False)):
                break
    print()
    print('Epoch:', i)
    print('Reward:', episode_rewards[i * k + ii])
    print('Eps:', env.eps)
    eps_history.append(env.eps)
    if (len(accurancy) > 0):
        print('accurancy:', accurancy[len(accurancy) - 1])
        print('loss:', loss[len(loss) - 1])

scale = 100
scale1 = 2000
moving_avg = np.convolve(episode_rewards, np.ones((scale,)) / scale, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.plot(episode_rewards)
plt.ylabel(f"Reward ma")
plt.xlabel("episode #")
plt.show()


# Plot results
accurancy = np.array(accurancy) * 100
accurancy = accurancy.tolist()
moving_avg1 = np.convolve(accurancy, np.ones((scale1,)) / scale1, mode='valid')
plt.plot([i for i in range(len(moving_avg1))], moving_avg1)
# plt.plot(episode_rewards)
plt.ylabel(f"Accurancy")
plt.xlabel("episode #")
plt.show()

moving_avg2 = np.convolve(loss, np.ones((scale1,)) / scale1, mode='valid')
plt.plot([i for i in range(len(moving_avg2))], moving_avg2)
# plt.plot(episode_rewards)
plt.ylabel(f"loss")
plt.xlabel("episode #")
plt.show()

moving_avg3 = np.convolve(eps_history, np.ones((scale,)) / scale, mode='valid')
plt.plot([i for i in range(len(moving_avg3))], moving_avg3)
# plt.plot(episode_rewards)
plt.ylabel(f"epsilone")
plt.xlabel("episode #")
plt.show()

env.reset()
fig = plt.figure()
obj, = plt.plot([], [], 'ro')
obj1, = plt.plot([], [], 'yo')
obj2, = plt.plot([], [], 'bo')
anim(plt)
