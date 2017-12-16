import numpy as np
import scipy
import tensorflow as tf
from agent_dir.agent import Agent
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from os.path import exists

OPT = 'adam'
LR = 0.0005
PRE = '2'

NAME = 'pg.a2c.nstep.pong.{0}.{1}.{2}'.format(PRE, OPT, LR)

# For testing
NAME = 'pg.a2c.pong.h5'
PRE = '2'

MODEL_NAME = '{0}.h5'.format(NAME)
LOG_NAME = '{0}.log'.format(NAME)

if OPT == 'adam': OPTIMIZER = Adam
elif OPT == 'rmsprop': OPTIMIZER = RMSprop

def policy_loss(advantage=0.0, beta=0.01):
    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=1) + K.epsilon()) * K.flatten(advantage)) + beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))
    return loss

def value_loss():
    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))
    return loss

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        self.state_size = (80, 80, 1)
        self.action_size = 3 # self.env.action_space.n
        self.value_size = 1
        self.gamma = 0.99
        self.learning_rate = LR
        
        # Training Memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Limit the memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        set_session(tf.Session(config=config))

        # Build the model
        _, _, self.a2c_networks, self.advantage = self.build_models()
        self.a2c_networks.compile(optimizer=OPTIMIZER(lr=LR), loss=[value_loss(), policy_loss(self.advantage, 0.01)])

        if args.test_pg_a2c or exists(MODEL_NAME):
            print('loading trained model')
            self.load(MODEL_NAME)

        if args.test_pg_a2c:
            np.random.seed(0)

        if PRE == '1':
            self.preprocess = self.preprocess_1
        elif PRE == '2':
            self.preprocess = self.preprocess_2

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.state = self.env.reset()
        self.prev_x = None
        self.score = 0
        self.steps = 0
        
    def build_models(self):
        state = Input(shape=self.state_size)
        C1 = Conv2D(32, (8, 8), strides=(4, 4), padding='same', 
                    activation='relu', kernel_initializer='he_normal')(state)
        C2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', 
                    activation='relu', kernel_initializer='he_normal')(C1)
        F = Flatten()(C2)
        D = Dense(128, activation='relu', kernel_initializer='he_normal')(F)
        value = Dense(1, activation='linear', kernel_initializer='he_normal')(D)
        policy = Dense(self.action_size, activation='softmax', kernel_initializer='he_normal')(D)
        
        value_network = Model(inputs=state, outputs=value)
        policy_network = Model(inputs=state, outputs=policy)
        
        advantage = Input(shape=(1, ))
        a2c_networks = Model(inputs=[state, advantage], outputs=[value, policy])
        a2c_networks.summary()
        
        return value_network, policy_network, a2c_networks, advantage
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    def act(self, observation, test=False):
        prob = self.a2c_networks.predict([observation, np.zeros((1, ))], batch_size=1)[1].flatten()
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action+1, prob

    def preprocess_1(self, I):
        I = I[35:195]                     # remove upper and lower parts
        I = I[::2, ::2, 0]                # pooling and only take the red channel
        I[I == 144] = I[I == 109] = 0     # remove the background
        I[I != 0] = 1                     # binarizing
        return I.astype(np.float).reshape(self.state_size)

    def preprocess_2(self, I):
        I = I[35:195]
        I = 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]
        I = I.astype(np.uint8)
        I = scipy.misc.imresize(I, [80, 80])
        return I.astype(np.float).reshape(self.state_size)
    
    def preprocess_state(self, state):
        curr_x = self.preprocess(state)
        x = curr_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
        self.prev_x = curr_x
        return np.expand_dims(x, axis=0)
    
    def train_networks(self):
        unroll = np.arange(self.steps)
        
        states = np.stack(self.states, axis=0).reshape((-1, 80, 80, 1))
        actions = np.stack(self.actions, axis=0)
        rewards = self.discount_rewards(np.stack(self.rewards, axis=0))
        # print(states.shape, actions.shape, rewards.shape)
        
        
        value, policy = self.a2c_networks.predict([states, unroll])
        
        advantages = rewards - value.ravel()
        targets = np.zeros((self.steps, self.action_size))
        targets[unroll, actions-1] = 1.0
        
        _, value_loss, policy_loss = self.a2c_networks.train_on_batch([states, advantages], [rewards, targets])
        entropy = np.mean(-policy * np.log(policy + 1e-8))
        
        self.states, self.actions, self.rewards = [], [], []
        
        return value_loss, policy_loss, entropy

    def train(self):
        """
        Implement your training algorithm here
        """
        self.init_game_setting()
        self.episode = 0
        
        while True:
            x = self.preprocess_state(self.state)
            action, prob = self.act(x)
            
            self.state, reward, done, _ = self.env.step(action)
            
            self.remember(x, action, reward)
            self.score += reward
            self.steps += 1
            
            if done:
                self.episode += 1
                value_loss, policy_loss, entropy = self.train_networks()
                print_and_log('Episode: {0} - Score: {1} - Step: {2} - Value Loss: {3} - Policy Loss: {4} - Entropy: {5}.'.format(
                    self.episode, self.score, self.steps, value_loss, policy_loss, entropy))
                
                self.init_game_setting()
                
                if self.episode % 10 == 0:
                    print('saving trained model')
                    self.save(MODEL_NAME)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        x = self.preprocess_state(observation)
        action, _ = self.act(x, test)
        return action
    
    def save(self, name):
        self.a2c_networks.save_weights(name)
    
    def load(self, name):
        self.a2c_networks.load_weights(name)

def print_and_log(msg):
    print(msg)
    with open(LOG_NAME, 'a') as f:
        f.write(msg + '\n')
