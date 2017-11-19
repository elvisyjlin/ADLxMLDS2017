import numpy as np
import tensorflow as tf
import sys
from os.path import join
from hw2_utils import MSVD, Predictions, word_embedding_loader
from hw2_s2vt import S2VT

if len(sys.argv) != 4:
	assert False, 'Error: invalid number of arguments. Predicting should be [hw2.py data_path model_file prediction_file]'

dataset_path = sys.argv[1]

x_dim = 4096
x_max_length = 80
y_max_length = 20
word_encoding_threshold = 1

num_units = 256
epochs = 4000
batch_size = 50
optimizer = 'rmsprop' # 'gd', 'adam', or 'rmsprop'
learning_rate = 0.001
max_to_keep = 50
random_every_epoch = True
shuffle_training_data = True

rnn_type = 'gru' # 'lstm' or 'gru'
use_dropout = None # None or a float number (dropout_rate)
use_attention = True # Fasle or True
use_scheduled = False # False or True
sampling_decaying_rate = None # a float between 0~1 e.g. 0.99
sampling_decaying_mode = None # 'linear or 'exponential'
sampling_decaying_per_epoch = None # an integer
use_embedding = 'fasttext' # None, 'word2vec', 'glove', or 'fasttext'

model_folder = 'models'
model_name = sys.argv[2]
model_file = join(model_folder, model_name)

prediction_file = sys.argv[3]
save_per_epoch = 50
load_from_epoch = 1049

print('# {}'.format(model_name))

msvd = MSVD(dataset_path, y_max_length, word_encoding_threshold)
init_bias_vector = msvd.get_bias_vector()
predictions = Predictions(msvd, dataset_path)
num_vocal = len(msvd.sentenceEncoder.word2int)
word_list = msvd.get_word_list()

s2vt = S2VT(x_dim, num_vocal, num_units, x_max_length, y_max_length, 
            batch_size, optimizer, learning_rate, rnn_type, 
            use_dropout, use_attention, use_scheduled, use_embedding, 
            word_list, y_bias_vector=init_bias_vector)
tf_embeds, tf_probs, tf_preds, tf_x, tf_x_seq_len, tf_x_max_len, tf_debug = s2vt.build_model_test()

saver = tf.train.Saver(max_to_keep=max_to_keep)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
print('Session started')

msvd.load_testing_data()

print('Testing started.')

model_file = join('models', '{}.ckpt'.format(model_name))
saver.restore(sess, model_file)
for x_, x_seq_len_, x_id in msvd.testing_data(batch_size):
    embeds, probs, preds, d = sess.run(
        [tf_embeds, tf_probs, tf_preds, tf_debug], 
        feed_dict={tf_x:         x_, 
                   tf_x_seq_len: x_seq_len_, 
                   tf_x_max_len: x_max_length})
    predictions.print(preds, False, True, '=> {}')
    predictions.add(x_id, preds)
predictions.save(prediction_file)

print('Finished predicting.')
