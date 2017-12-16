import numpy as np
import scipy
import tensorflow as tf
from agent_dir.agent import Agent
from keras.backend.tensorflow_backend import set_session
from keras.layers import Reshape, Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from os.path import exists

OPT = 'rmsprop'
LR = 0.00005
PRE = '2'
LOSS = 'categorical_crossentropy' # 'categorical_crossentropy' or 'mse'

NAME = 'pg.pong.{0}.{1}.{2}.{3}'.format(PRE, OPT, LR, LOSS)

# For testing
NAME = 'pg.pong'
PRE = '2'

MODEL_NAME = '{0}.h5'.format(NAME)
LOG_NAME = '{0}.log'.format(NAME)

if OPT == 'adam': OPTIMIZER = Adam
elif OPT == 'rmsprop': OPTIMIZER = RMSprop

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        self.state_size = (80, 80, 1)
        self.action_size = 3 # self.env.action_space.n
        self.gamma = 0.99
        self.learning_rate = LR
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        
        # Limit the memory usage
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # set_session(tf.Session(config=config))

        # Build the model
        self.model = self.build_model()

        if args.test_pg or exists(MODEL_NAME):
            print('loading trained model')
            self.load(MODEL_NAME)

        if args.test_pg:
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
        
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', 
                         activation='relu', kernel_initializer='he_normal', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', 
                         activation='relu', kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_normal'))
        opt = OPTIMIZER(lr=self.learning_rate)
        model.compile(loss=LOSS, optimizer=opt)
        model.summary()
        return model
    
    def remember(self, state, action, prob, reward):
        action -= 1
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(y.astype(np.float) - prob)
        self.states.append(state)
        self.rewards.append(reward)
    
    def act(self, observation, test=False):
        observation = np.expand_dims(observation, axis=0)
        aprob = self.model.predict(observation, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        action += 1
        return action, prob
        
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

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
        return x
    
    def train_model(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards) # normalization?
        # rewards = rewards / np.std(rewards)
        gradients *= rewards
        
        X = np.vstack([self.states])
        Y = self.probs + self.learning_rate * np.vstack([gradients])
        step_size = X.shape[0]
        loss = self.model.train_on_batch(X, Y)
        # self.model.fit(x=X, y=Y, batch_size=1024, epochs=1, verbose=1)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        return step_size, loss

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
            self.score += reward
            self.remember(x, action, prob, reward)
            
            if done:
                self.episode += 1
                step_size, loss = self.train_model()
                print_and_log('Episode: {0} - Score: {1} - Step: {2} - Loss: {3}.'.format(
                    self.episode, self.score, step_size, loss))
                
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
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)

def print_and_log(msg):
    print(msg)
    with open(LOG_NAME, 'a') as f:
        f.write(msg + '\n')
