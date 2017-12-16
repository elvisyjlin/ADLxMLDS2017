import random
import numpy as np
import tensorflow as tf
from agent_dir.agent import Agent
from collections import deque
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from os.path import exists
from skimage.color import rgb2gray

QN = 'dqn'
DUEL = 'avg' # 'none', 'avg', 'max', or 'naive'
OPT = 'adam'
LR = 0.0001
FINAL_EPI = 0.05
TEST_EPI = 0.01

NAME = 're.{0}.{1}.breakout.{2}.{3}.{4}'.format(QN, DUEL, OPT, LR, FINAL_EPI) # 'ddqn.breakout.adam'

NAME = 'ddqn.max.breakout'
QN = 'ddqn'
DUEL = 'max'
OPT = 'adam'

NETWORK_NAME = '{0}.h5'.format(NAME)
LOG_NAME = '{0}.log'.format(NAME)

if OPT == 'adam': OPTIMIZER = Adam
elif OPT == 'rmsprop': OPTIMIZER = RMSprop

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        
        self.t = 0

        self.state_size = (84, 84)
        self.state_length = 4
        self.model_input_shape = (84, 84, self.state_length)
        self.action_size = self.env.action_space.n
        self.memory = deque()
        self.memory_size = 10000
        
        self.gamma = 0.99
        self.epsilon_init = 1.0
        self.epsilon_min = FINAL_EPI
        self.exploration_steps = 1000000
        self.epsilon_step = (self.epsilon_init - self.epsilon_min) / self.exploration_steps
        self.epsilon = self.epsilon_init
        
        self.learning_rate = LR
        
        self.initial_replay_size = 10000
        self.replay_interval = 4
        self.target_update_interval = 1000
        self.save_interval = 50000
        
        # For summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0
        
        # Limit the memory usage
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # set_session(tf.Session(config=config))
        
        # Q network and Target network
        self.main_network = self.build_network()
        self.target_network = self.build_network()

        if args.test_dqn or exists(NETWORK_NAME):
            print('loading trained model {0}'.format(NETWORK_NAME))
            self.load(NETWORK_NAME)
        
        self.update_target_network()

        if args.test_dqn:
            np.random.seed(0)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.prev_state = None
        self.state = self.env.reset()
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        
    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), border_mode='same', activation='relu', input_shape=self.model_input_shape))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), border_mode='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), border_mode='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # model.add(Dense(512))
        # model.add(LeakyReLU())
        if DUEL == 'none':
            model.add(Dense(self.action_size, activation='linear'))
        else:
            model.add(Dense(self.action_size + 1, activation='linear'))
            if DUEL == 'avg':
                model.add(Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(
                    a[:, 1:], keepdims=True), output_shape=(self.action_size, )))
            elif DUEL == 'max':
                model.add(Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.max(
                    a[:, 1:], keepdims=True), output_shape=(self.action_size, )))
            elif DUEL == 'naive':
                model.add(Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:], output_shape=(self.action_size, )))
        opt = OPTIMIZER(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        
    def act(self, state):
        if np.random.rand() <= self.epsilon or self.t < self.initial_replay_size:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.main_network.predict(np.expand_dims(state, axis=0), batch_size=1)[0])
        
        if self.epsilon > self.epsilon_min and self.t >= self.initial_replay_size:
            self.epsilon -= self.epsilon_step
        
        return action
    
    def replay(self, batch_size):
        
        # Initialize batches
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        Y = []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = np.stack(states)           # (32, 84, 84, 4)
        actions = np.stack(actions)         # (32, )
        rewards = np.stack(rewards)         # (32, )
        next_states = np.stack(next_states) # (32, 84, 84, 4)
        dones = np.array(dones) + 0         # (32, )
        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        
        # About the next state
        target_q_values = self.target_network.predict(next_states, batch_size=batch_size) # (32, 4)
        if QN == 'dqn':
            max_target_q_values = np.amax(target_q_values, axis=1)                        # (32, )
        elif QN == 'ddqn':
            q_values = self.main_network.predict(next_states, batch_size=batch_size)      # (32, 4)
            best_actions = np.argmax(q_values, axis=1)                                    # (32, )
            max_target_q_values = target_q_values[range(batch_size), best_actions]        # (32, )
        max_target_q_values = rewards + (1 - dones) * self.gamma * max_target_q_values    # (32, )
        
        # About the current state
        target_f = self.main_network.predict(states, batch_size=batch_size)
        target_f[range(batch_size), actions] = max_target_q_values
        
        # Train the main network
        hist = self.main_network.fit(states, target_f, batch_size=batch_size, epochs=1, verbose=0)
        self.total_loss += hist.history['loss'][0]
        
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def train(self):
        """
        Implement your training algorithm here
        """
        self.init_game_setting()
        while True:
            action = self.act(self.state)
            next_state, reward, done, _ = self.env.step(action)
            self.remember(self.state, action, reward, next_state, done)
            self.state = next_state
            
            if self.t >= self.initial_replay_size:
                if self.t % self.replay_interval == 0:
                    self.replay(32)
                if self.t % self.target_update_interval == 0:
                    self.update_target_network()
                if self.t % self.save_interval == 0:
                    print('saving trained model {0}'.format(NETWORK_NAME))
                    self.save(NETWORK_NAME)
                
            self.total_reward += reward
            self.total_q_max += np.amax(self.main_network.predict(np.expand_dims(next_state, axis=0), batch_size=1)[0])
            self.duration += 1
            
            if done:
                self.episode += 1
                
                if self.t < self.initial_replay_size:
                    mode = 'random'
                elif self.initial_replay_size <= self.t < self.initial_replay_size + self.exploration_steps:
                    mode = 'explore'
                else:
                    mode = 'exploit'
                    
                print_and_log('Episode: {0} - Time Step: {1} - Duration: {2} - Epsilon: {3} - Total Reward: {4} - Avg Q Max: {5} - Avg Loss: {6} - Mode: {7}.'.format(
                    self.episode, self.t, self.duration, self.epsilon, 
                    self.total_reward, self.total_q_max / float(self.duration), 
                    self.total_loss / (float(self.duration)/self.replay_interval), mode))
                
                self.init_game_setting()
            
            self.t += 1

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        if np.random.rand() <= TEST_EPI:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.main_network.predict(np.expand_dims(observation, axis=0), batch_size=1)[0])
        self.t += 1
        return action
    
    def save(self, name):
        self.main_network.save_weights(name)
    
    def load(self, name):
        self.main_network.load_weights(name)

def print_and_log(msg):
    print(msg)
    with open(LOG_NAME, 'a') as f:
        f.write(msg + '\n')
