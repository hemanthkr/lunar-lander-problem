import numpy as np

import gym

import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import random
from collections import deque
import time
#tf.disable_v2_behavior() # testing on tensorflow 1

class Agent:
    def __init__(self, env, optimizer, batch_size):
        # general info
        self.state_size = env.observation_space.shape[0] # number of factors in the state; e.g: velocity, position, etc
        self.action_size = env.action_space.n
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        # allow large replay exp space
        self.replay_exp = deque(maxlen=1000000)
        
        self.gamma = 0.99
        self.epsilon = 1.0 # initialize with high exploration, which will decay later
        
        # Build Policy Network
        self.brain_policy = Sequential()
        self.brain_policy.add(Dense(128, input_dim = self.state_size, activation = "relu"))
        self.brain_policy.add(Dense(128 , activation = "relu"))
        self.brain_policy.add(Dense(self.action_size, activation = "linear"))
        self.brain_policy.compile(loss = "mse", optimizer = self.optimizer)
        
        print("Brain Policy")
        print('*' * 40)
        print(self.brain_policy.summary())
        print('*' * 40)
        
        # Build Target Network
        self.brain_target = Sequential()
        self.brain_target.add(Dense(128, input_dim = self.state_size, activation = "relu"))
        self.brain_target.add(Dense(128 , activation = "relu"))
        self.brain_target.add(Dense(self.action_size, activation = "linear"))
        self.brain_target.compile(loss = "mse", optimizer = self.optimizer)
        
        print("Brain Target")
        print("*" * 40)
        print(self.brain_target.summary())
        print("*" * 40)
        
        self.update_brain_target()
    
    # add new experience to the replay exp
    def memorize_exp(self, state, action, reward, next_state, done):
        self.replay_exp.append((state, action, reward, next_state, done))
    
    def update_brain_target(self):
        return self.brain_target.set_weights(self.brain_policy.get_weights())
    
    def choose_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon: # exploration
            action = np.random.choice(self.action_size)
        else:
            state = np.reshape(state, [1, state_size])
            qhat = self.brain_policy.predict(state) # output Q(s,a) for all a of current state
            action = np.argmax(qhat[0]) # because the output is m * n, so we need to consider the dimension [0]
            
        return action
     
    # update params in NN
    def learn(self):

        
        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.replay_exp), self.batch_size)
        mini_batch = random.sample(self.replay_exp, cur_batch_size)
        
        # batch data
        sample_states = np.ndarray(shape = (cur_batch_size, self.state_size)) # replace 128 with cur_batch_size

        sample_actions = np.ndarray(shape = (cur_batch_size, 1))
        sample_rewards = np.ndarray(shape = (cur_batch_size, 1))
        sample_next_states = np.ndarray(shape = (cur_batch_size, self.state_size))
        sample_dones = np.ndarray(shape = (cur_batch_size, 1))
        
        temp=0
        for exp in mini_batch:
            sample_states[temp] = exp[0]
            sample_actions[temp] = exp[1]
            sample_rewards[temp] = exp[2]
            sample_next_states[temp] = exp[3]
            sample_dones[temp] = exp[4]
            temp += 1
        
         
        sample_qhat_next = self.brain_target.predict(sample_next_states)
        
        
        # set all Q values terminal states to 0
        sample_qhat_next = sample_qhat_next * (np.ones(shape = sample_dones.shape) - sample_dones)
        # choose max action for each state
        sample_qhat_next = np.max(sample_qhat_next, axis=1)
        
        sample_qhat = self.brain_policy.predict(sample_states)
        
        for i in range(cur_batch_size):
            a = sample_actions[i,0]
            sample_qhat[i,int(a)] = sample_rewards[i] + self.gamma * sample_qhat_next[i]
            
        q_target = sample_qhat
            
        self.brain_policy.fit(sample_states, q_target, epochs = 1, verbose = 0)
        
env = gym.make("LunarLander-v2")
optimizer = Adam(learning_rate = 0.0001)

agent = Agent(env, optimizer, batch_size = 64)
state_size = env.observation_space.shape[0]

#state = env.reset()

#print("shape is ",env.observation_space.shape[0])

# load model
#agent.brain_policy.set_weights(tf.keras.models.load_model('DQN_Model.h5').get_weights())

timestep=0
rewards = []
aver_reward = []
aver = deque(maxlen=100)


for episode in range(500):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        
        env.render()

        total_reward += reward
        
        agent.memorize_exp(state, action, reward, next_state, done)
        agent.learn()
        
        state = next_state
        timestep += 1
        
        
    aver.append(total_reward)     
    aver_reward.append(np.mean(aver))
    
    rewards.append(total_reward)
    
    # update model_target after each episode
   # if episode % 5==0:
        #print("Training step is ",episode)
        
    agent.update_brain_target()
    agent.epsilon = max(0.1, 0.995 * agent.epsilon) # decaying exploration
    print("Episode ", episode, total_reward)
    
   
    
#plt.title("Learning Curve")
#plt.xlabel("Episode")
#plt.ylabel("Reward")
#plt.plot(rewards)
#plt.show()

plt.title("Agent Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(rewards,label ='rewards' )
plt.plot(aver_reward,label ='average rewards' )
plt.legend()
plt.show()

agent.brain_policy.save('DQN_Model.h5')

    