import numpy as np
from glob import glob
from os import mkdir
from os.path import join, basename, exists

LOAD_MODEL = 'pretrained_model'
OUTPUT_PATH = 'samples'
if not exists(OUTPUT_PATH): mkdir(OUTPUT_PATH)

REDUCE_DATASET = True
USE_BN_D = False
USE_BN_G = True
WEIGHTED_LOSS = True
WRONG_LABELS_ARE_ZEROS = True
RANDOM_NOISE_PER_TRAIN = False

HW4_HAIR_TAGS = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
                 'green hair', 'red hair', 'purple hair', 'pink hair',
                 'blue hair', 'black hair', 'brown hair', 'blonde hair']

HW4_EYES_TAGS = ['gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

import keras.backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')
K.set_learning_phase(True)
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Lambda
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal, TruncatedNormal

conv_init = TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
gamma_init = RandomNormal(1., 0.02)

class C_DCGAN:
    def __init__(self, isize, nz, ne, nc, ndf, ngf):
        self.isize = isize
        self.nz = nz
        self.ne = ne
        self.nc = nc
        self.ndf = ndf
        self.ngf = ngf
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        set_session(tf.Session(config=config))
        
        self.discriminator_network = self.build_discriminator()
        self.generator_network = self.build_generator()
    def build_discriminator(self):
        isize, nz, ne, nc, ndf = self.isize, self.nz, self.ne, self.nc, self.ndf
        i_i = inputs_i = Input(shape=(isize, isize, nc))
        i_t = inputs_t = Input(shape=(ne,))
        i_t = Lambda(quad)(i_t)
        _ = Conv2D(filters=ndf, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(i_i)
        _ = LeakyReLU(0.2)(_)
        _ = Conv2D(filters=ndf*2, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(_)
        if USE_BN_D: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = LeakyReLU(0.2)(_)
        _ = Conv2D(filters=ndf*4, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(_)
        if USE_BN_D: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = LeakyReLU(0.2)(_)
        _ = Conv2D(filters=ndf*8, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(_)
        if USE_BN_D: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = LeakyReLU(0.2)(_)
        _ = Concatenate(axis=-1)([_, i_t])
        _ = Conv2D(filters=ndf*8, kernel_size=1, strides=1, padding='same', kernel_initializer=conv_init)(_)
        if USE_BN_D: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = LeakyReLU(0.2)(_)
        _ = Flatten()(_)
        outputs = Dense(1, kernel_initializer=conv_init)(_)
        return Model(inputs=[inputs_i, inputs_t], outputs=outputs)
    def build_generator(self):
        isize, nz, ne, nc, ngf = self.isize, self.nz, self.ne, self.nc, self.ngf
        i_z = inputs_z = Input(shape=(nz,))
        i_t = inputs_t = Input(shape=(ne,))
        _ = Concatenate(axis=-1)([i_z, i_t])
        _ = Dense(ngf*8 * 4 * 4, kernel_initializer=conv_init)(_)
        _ = Reshape((4, 4, ngf*8))(_)
        if USE_BN_G: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = Activation('relu')(_)
        _ = Conv2DTranspose(filters=ngf*4, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(_)
        if USE_BN_G: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = Activation('relu')(_)
        _ = Conv2DTranspose(filters=ngf*2, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(_)
        if USE_BN_G: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = Activation('relu')(_)
        _ = Conv2DTranspose(filters=ngf*1, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(_)
        if USE_BN_G: _ = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)(_)
        _ = Activation('relu')(_)
        _ = Conv2DTranspose(filters=nc, kernel_size=5, strides=2, padding='same', kernel_initializer=conv_init)(_)
        outputs = Activation('tanh')(_)
        return Model(inputs=[inputs_z, inputs_t], outputs=outputs)
    def discriminator(self):
        return self.discriminator_network
    def generator(self):
        return self.generator_network
    def load(self, name, direct=False):
        self.load_discriminator('{0}.D.h5'.format(name), direct)
        self.load_generator('{0}.G.h5'.format(name), direct)
    def load_discriminator(self, name, direct=False):
        if direct:
            self.discriminator_network.load_weights(name)
        else:
            self.discriminator_network.load_weights(join(OUTPUT_PATH, name))
    def load_generator(self, name, direct=False):
        if direct:
            self.generator_network.load_weights(name)
        else:
            self.generator_network.load_weights(join(OUTPUT_PATH, name))
    def save(self, name, direct=False):
        self.save_discriminator('{0}.D.h5'.format(name), direct)
        self.save_generator('{0}.G.h5'.format(name), direct)
    def save_discriminator(self, name, direct=False):
        if direct:
            self.discriminator_network.save_weights(name)
        else:
            self.discriminator_network.save_weights(join(OUTPUT_PATH, name))
    def save_generator(self, name, direct=False):
        if direct:
            self.generator_network.save_weights(name)
        else:
            self.generator_network.save_weights(join(OUTPUT_PATH, name))
def quad(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=2)
    x = K.tile(x, [1, 4, 4, 1])
    return x

nc = 3
nz = 100
ne = 23
ngf = 64
ndf = 64

imageSize = 64
batchSize = 64
lr, beta_1, beta_2 = 2e-4, 0.5, 0.99

gan = C_DCGAN(imageSize, nz, ne, nc, ndf, ngf)

if LOAD_MODEL is not None:
    gan.load(LOAD_MODEL, direct=True)

netG = gan.generator()

from PIL import Image
# from IPython.display import display

def parse_tags(tags):
    hair = np.zeros(len(HW4_HAIR_TAGS))
    for idx, tag in enumerate(HW4_HAIR_TAGS):
        if tag in tags:
            hair[idx] = 1
    eyes = np.zeros(len(HW4_EYES_TAGS))
    for idx, tag in enumerate(HW4_EYES_TAGS):
        if tag in tags:
            eyes[idx] = 1
    return np.concatenate([hair, eyes])

def show_and_save(image, rows=1, save=None):
    assert image.shape[0]%rows == 0
    image = ((image+1)/2*255).clip(0,255).astype('uint8')
    image = image.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    image = Image.fromarray(image)
    # display(image)
    if save is not None: image.save(save)

from sys import argv
TEST = False
if TEST or len(argv) == 1:
    sample_noise = np.random.normal(size=(sample_size, nz)).astype('float32')
    samples = netG.predict([sample_noise, sample_text])
    show_and_save(samples, sample_rows)
else:
    fixed_noise = np.load('fixed_noise.npy')
    testing_text_file = argv[1]
    with open(testing_text_file, 'r') as f:
        for line in f.readlines():
            id, tags = line.strip().split(',')
            text = parse_tags(tags).reshape((1, ne))
            for i in range(5):
                sample = netG.predict([fixed_noise[i:i+1], text])
                show_and_save(sample, 1, save=join(OUTPUT_PATH, 'sample_{0}_{1}.jpg'.format(id, i+1)))
