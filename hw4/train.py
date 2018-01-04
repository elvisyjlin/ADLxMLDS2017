import numpy as np
from glob import glob
from os import mkdir
from os.path import join, basename, exists
from sys import argv

OUTPUT_PATH = 'output'
LOAD_MODEL = None

REDUCE_DATASET = True
USE_BN_D = False
USE_BN_G = True
WEIGHTED_LOSS = True
WRONG_LABELS_ARE_ZEROS = True
RANDOM_NOISE_PER_TRAIN = False

OUTPUT_PATH = OUTPUT_PATH.format(REDUCE_DATASET, USE_BN_D, USE_BN_G, WEIGHTED_LOSS, 
                                 WRONG_LABELS_ARE_ZEROS, RANDOM_NOISE_PER_TRAIN)
if not exists(OUTPUT_PATH): mkdir(OUTPUT_PATH)

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
# lr, beta_1, beta_2 = 1e-4, 0.5, 0.99
lr, beta_1, beta_2 = 2e-4, 0.5, 0.99
# lr, beta_1, beta_2 = 5e-5, 0.5, 0.9

gan = C_DCGAN(imageSize, nz, ne, nc, ndf, ngf)

if LOAD_MODEL:
    gan.load_discriminator('{0}.D.h5'.format(LOAD_MODEL), direct=False)
    gan.load_generator('{0}.G.h5'.format(LOAD_MODEL), direct=False)

netD = gan.discriminator()
netD.summary()
netG = gan.generator()
netG.summary()

from keras.optimizers import RMSprop, SGD, Adam

right_text_input = Input(shape=(ne,))
wrong_text_input = Input(shape=(ne,))

real_data_input = Input(shape=(imageSize, imageSize, nc))
wrong_data_input = Input(shape=(imageSize, imageSize, nc))
noise_input = Input(shape=(nz,))
fake_data_input = netG([noise_input, right_text_input])

real_right_output = netD([real_data_input, right_text_input])
fake_right_output = netD([fake_data_input, right_text_input])
real_wrong_output = netD([real_data_input, wrong_text_input])
wrong_right_output = netD([wrong_data_input, right_text_input])

real_right_loss = K.mean(K.binary_crossentropy(
    K.ones_like(real_right_output), K.sigmoid(real_right_output), from_logits=False), axis=-1)
fake_right_loss = K.mean(K.binary_crossentropy(
    K.zeros_like(fake_right_output), K.sigmoid(fake_right_output), from_logits=False), axis=-1)
real_wrong_loss = K.mean(K.binary_crossentropy(
    K.zeros_like(real_wrong_output), K.sigmoid(real_wrong_output), from_logits=False), axis=-1)
wrong_right_loss = K.mean(K.binary_crossentropy(
    K.zeros_like(wrong_right_output), K.sigmoid(wrong_right_output), from_logits=False), axis=-1)

if WEIGHTED_LOSS:
    loss = real_right_loss + (fake_right_loss + real_wrong_loss + wrong_right_loss) / 3
else:
    loss = real_right_loss + fake_right_loss + real_wrong_loss + wrong_right_loss

training_updates = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2).get_updates(netD.trainable_weights, [], loss)
netD_train = K.function([real_data_input, wrong_data_input, noise_input, right_text_input, wrong_text_input],
                        [real_right_loss, fake_right_loss, real_wrong_loss, wrong_right_loss],    
                        training_updates)

loss = K.mean(K.binary_crossentropy(
    K.ones_like(fake_right_output), K.sigmoid(fake_right_output), from_logits=False), axis=-1)

training_updates = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2).get_updates(netG.trainable_weights, [], loss)
netG_train = K.function([noise_input, right_text_input], [loss], training_updates)

fixed_rows = 11
fixed_cols = 12
fixed_size = fixed_rows * fixed_cols
fixed_noise = np.random.normal(size=(fixed_size, nz)).astype('float32')
fixed_text = np.zeros((fixed_size, ne))
for i in range(fixed_rows):
    for j in range(fixed_cols):
        fixed_text[i*fixed_cols+j][j] = 1
        fixed_text[i*fixed_cols+j][fixed_cols+i] = 1

from PIL import Image
import numpy as np
import glob
from random import randint, shuffle

def load_data(file_pattern):
    return np.array(sorted(glob.glob(file_pattern), key=lambda x: int(basename(x).split('.')[0])), dtype=np.str)

def load_labels():
    return np.load('onehot_labels.npy')

def read_image(fn, image_size=64):
    im = Image.open(fn).convert('RGB')
    im = im.resize((image_size, image_size), Image.BILINEAR)
    img = np.array(im)/255*2-1
    if randint(0,1):
        img=img[:,::-1]
    return img

def filter_data(data, labels, num_max=2):
    rows, cols = labels.shape
    indice = [row for row in range(rows) if any(
        labels[row][col] > 0.0 for col in range(cols)) and len(np.argwhere(labels[row] == 1.0)) <= num_max]
    data = data[indice]
    labels = labels[indice]
    return data, labels

def minibatch(data, batchsize, filter=False, epoch_start=0):
    labels = load_labels()
    if filter:
        data, labels = filter_data(data, labels)
    print('Training data size: {0}, training label size: {1}'.format(len(data), len(labels)))
    length = len(data)
    epoch = i = epoch_start
    tmpsize = None
    p = list(range(length))
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(p)
            data = data[p]
            labels = labels[p]
            i = 0
            epoch+=1
        rtn_X = [read_image(data[j]) for j in range(i,i+size)]
        rtn_y = labels[i:i+size]
        i+=size
        tmpsize = yield epoch, np.float32(rtn_X), rtn_y #, data[i:i+size]

train_data = load_data(join(argv[1], '*.jpg'))

from PIL import Image
# from IPython.display import display

def show_and_save(X, rows=1, save=None):
    assert X.shape[0]%rows == 0
    int_X = ((X+1)/2*255).clip(0,255).astype('uint8')
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    image = Image.fromarray(int_X)
    # display(image)
    if save is not None: image.save(join(OUTPUT_PATH, save))

def print_and_log(msg):
    print(msg)
    with open(join(OUTPUT_PATH, 'output.logs'), 'a') as f:
        f.write('{0}\n'.format(msg))

train_batch = minibatch(train_data, fixed_size, filter=True)
_, X, _ = next(train_batch)
show_and_save(X, rows=fixed_rows, save='sample.jpg')
del train_batch, X

import time
t0 = time.time()
niter = 500
Diters = 1
Giters = 2
show_every_gen_iter = 100
save_every_gen_iter = 1000
errG = 0
targetD = np.float32([2]*batchSize+[-2]*batchSize)[:, None]
targetG = np.ones(batchSize, dtype=np.float32)[:, None]
epoch, gen_iterations = 0, 0
# if LOAD_MODEL:
#     if '/' in LOAD_MODEL:
#         epoch = int(LOAD_MODEL.split('/')[1].split('-')[0]) + 1
#     else:
#         epoch = int(LOAD_MODEL.split('-')[0]) + 1
train_batch = minibatch(train_data, batchSize, REDUCE_DATASET, epoch)
while epoch < niter:
    #  每個 epoch 洗牌一下
    epoch, train_X, train_y = next(train_batch)
    if epoch >= niter: break
    
    if WRONG_LABELS_ARE_ZEROS:
        train_X_wrong, train_y_wrong = np.roll(train_X, 1, axis=0), np.zeros_like(train_y)
    else:
        train_X_wrong, train_y_wrong = np.roll(train_X, 1, axis=0), np.roll(train_y, 1, axis=0)
        
    if not RANDOM_NOISE_PER_TRAIN: noise = np.random.normal(size=(batchSize, nz))
        
    for _ in range(Diters):
        if RANDOM_NOISE_PER_TRAIN: noise = np.random.normal(size=(batchSize, nz))
        errD_real_right, errD_fake_right, errD_real_wrong, errD_wrong_right = netD_train(
            [train_X, train_X_wrong, noise, train_y, train_y_wrong])
        if WEIGHTED_LOSS:
            errD = errD_real_right + (errD_fake_right + errD_real_wrong + errD_wrong_right) / 3
        else:
            errD = errD_real_right + errD_fake_right + errD_real_wrong + errD_wrong_right
            
    for _ in range(Giters):
        if gen_iterations%show_every_gen_iter==0:
            print_and_log('[{:d}/{:d}][{:d}] Loss_D: {:f} Loss_G: {:f} Loss_D_rr: {:f} Loss_D_fr {:f} Loss_D_rw: {:f} Loss_D_wr {:f} Duration: {:f}'.format(
                epoch, niter, gen_iterations, np.mean(errD), np.mean(errG), 
                np.mean(errD_real_right), np.mean(errD_fake_right), np.mean(errD_real_wrong), np.mean(errD_wrong_right), time.time()-t0))
            fake = netG.predict([fixed_noise, fixed_text])
            show_and_save(fake, fixed_rows, save='{0}-{1}.jpg'.format(epoch, gen_iterations))
        if gen_iterations%save_every_gen_iter==0:
            gan.save(argv[2], direct=True)
            
        if RANDOM_NOISE_PER_TRAIN: noise = np.random.normal(size=(batchSize, nz))
        errG, = netG_train([noise, train_y])
        gen_iterations += 1
