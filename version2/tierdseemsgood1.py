import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adamax

import cv2

import gym
from gym import wrappers

sns.set_style("ticks")
sns.despine()

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

env = gym.make("CarRacing-v0")

env = wrappers.Monitor(env, "train_1", force=True, mode='training')
#env = wrappers.Monitor(env, "larger_image", resume=True, mode='training')
#env = wrappers.Monitor(env, "larger_image", resume=True, mode='evaluation')
def transform(observation_rgb):
    controlStateBar = observation_rgb[84:, 12:]
    observation_grayScale = cv2.cvtColor(controlStateBar, cv2.COLOR_RGB2GRAY)
    observation_bw = cv2.threshold(observation_grayScale, 1, 255, cv2.THRESH_BINARY)[1]#output has 3 eleman second is converted to 0,255 array
    observation_bw = observation_bw.astype('float')/255
    
    game_board = observation_rgb[:84, 6:90]
    observation_grayScale = cv2.cvtColor(game_board, cv2.COLOR_RGB2GRAY)
    game_board_bw = cv2.threshold(observation_grayScale, 120, 255, cv2.THRESH_BINARY)[1]
    game_board_bw = cv2.resize(game_board_bw, (10, 10), interpolation=cv2.INTER_NEAREST)
    game_board_bw = game_board_bw.astype('float')/255

    car = observation_rgb[66:78, 43:53]
    car_grayScale = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY)
    car_bw = cv2.threshold(car_grayScale, 90, 255, cv2.THRESH_BINARY)[1]
    # print car_bw
    car_field_t = [car_bw[:, 3].mean()/255, 
                   car_bw[:, 4].mean()/255,
                   car_bw[:, 5].mean()/255, 
                   car_bw[:, 6].mean()/255]
    # print car_field_t
    return observation_bw, game_board_bw, car_field_t

def compute_satate(observation_np):
	right_steering = observation_np[6, 36:46].mean()#kolan 85 tast, max range turning right 36:46
	left_steering = observation_np[6, 26:36].mean()#max range 26:36 az oon traf 
	steering = (right_steering - left_steering + 1.0)/2

	left_gyro = observation_np[6, 46:60].mean()
	right_gyro = observation_np[6, 60:76].mean()
	gyro = (right_gyro - left_gyro + 1.0)/2

	speed = observation_np[:, 0][:-2].mean()
	abs1 = observation_np[:, 6][:-2].mean()
	abs2 = observation_np[:, 8][:-2].mean()
	abs3 = observation_np[:, 10][:-2].mean()
	abs4 = observation_np[:, 12][:-2].mean()
	return [steering, speed, gyro, abs1, abs2, abs3, abs4]

vector_size = 10*10 + 7 + 4

def create_nn():
    if os.path.exists('race-car_larger.h5'):
        return load_model('race-car_larger.h5')
    
    model = Sequential()
    model.add(Dense(512, init='lecun_uniform', input_shape=(vector_size,)))  # 7x7+3 or 14x14+3
    model.add(Activation('relu'))
    
    model.add(Dense(11, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have a range of real-valued opts.
    
    model.compile(loss='mse', optimizer=Adamax())  # lr=0.001
    model.summary()
    
    return model

class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn()  # One FFNN for all actions
        
    def predict(self, s):
        return self.model.predict(s.reshape(-1, vector_size), verbose=0)[0]
    
    def update(self, s, G):
        self.model.fit(s.reshape(-1, vector_size), 
                       np.array(G).reshape(-1, 11), 
                       nb_epoch=1, 
                       verbose=0)
        
    def sample_action(self, s, eps):
        qval = self.predict(s)
        if np.random.random() < eps:
            return random.randint(0,10), qval
        else:
        	return np.argmax(qval), qval

def find_action(q_index):
    # We reduce the action space to 
    
    gas = 0.0
    brake = 0.0
    steering = 0.0
    
    # Output value ranges from 0 to 10:
    
    if q_index <= 8:
        # Steering, brake, and gas are zero
        q_index -= 4
        steering = float(q_index)/4
    elif q_index >=9 and q_index <=9:
        q_index -= 8
        gas = float(q_index)/3  # 33% of gas
    elif q_index >= 10 and q_index <= 10:
        q_index -= 9
        brake = float(q_index)/2  # 50% of brake
    else:
        print("Error")  #Why?
    return [steering, gas, brake]

def play_one(env, model, eps, gamma,doRender):
    observation = env.reset()
    done = False
    full_reward_received = False# game not ended
    totalreward = 0
    iters = 0
    while not done:
        observation_np, gameboard_np, car_np = transform(observation)
        aaa = compute_satate(observation_np)
        state = np.concatenate((np.array([aaa]).reshape(1,-1).flatten(),
                               gameboard_np.reshape(1, -1).flatten(),car_np), axis=0)  # 3+7*7 size vector, scaled in range 0-1
        argmax_qval, qval = model.sample_action(state, eps)
        prev_state = state
        action = find_action(argmax_qval)#choose action for next state depends on value state
        observation, reward, done, info = env.step(action)
        observation_np, gameboard_np, car_np = transform(observation)
        state = np.concatenate((np.array([compute_satate(observation_np)]).reshape(1,-1).flatten(),
                               gameboard_np.reshape(1,-1).flatten(), car_np), axis=0)
        # Update the model, standard Q-Learning TD(0)
        next_qval = model.predict(state)
        G = reward + gamma*np.max(next_qval)
        y = qval[:]
        y[argmax_qval] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1
        if doRender:
            env.render()
        if iters > 16000:
            print("This episode is stuck.")
            break
            
    return totalreward, iters

model = Model(env)
gamma = 0.99# change gama to

N = 100
totalrewards = np.empty(N)
costs = np.empty(N)

for n in range(N):
    eps = 0.5/np.sqrt(n+1+900)#change epsilon to 
    print eps,n
    doRender = False
    if (n+1)%1 == 0 :
        doRender = True
    totalreward, iters = play_one(env, model, eps, gamma,doRender)
    totalrewards[n] = totalreward
    print("Episode: ", n, 
          ", iters: ", iters, 
          ", total reward: ", totalreward, 
          ", epsilon: ", eps, 
          ", average reward (of last 100): ", totalrewards[max(0,n-100):(n+1)].mean()
         )
    # We save the model every 10 episodes:
    if n%1 == 0:
        model.model.save('race-car_larger.h5')
        
print("Average reward for the last 100 episodes: ", totalrewards[-100:].mean())
print("Total steps: ", totalrewards.sum())

plt.plot(totalrewards)
plt.title("Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

plot_running_avg(totalrewards)
model.model.save('race-car_test1.h5')
env.close()