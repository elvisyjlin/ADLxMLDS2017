
# coding: utf-8

# In[ ]:


import sys

# Environment Parameters

# mode = 'train'
# mode = 'load_train'
# mode = 'predict'
mode = 'load_predict'

# model_name = 'hw2_special'
model_name = 'models/hw2_special'
prediction_name = sys.argv[2]
# prediction_name = 'predictions_std_40_1'
iter_epochs = None
# iter_epochs = range(99, 1000, 100)

training_max_time_steps = 40
word_encoding_threshold = 1
random_every_epoch = True
shuffle_training_data = True
save_per_epoch = 100

num_units = 256
num_layers = 2
x_embedding_size = 4096

use_dropout = True
output_keep_prob = 0.5 if 'train' in mode else 1.0
use_residual = True 
projection_using_bias = False
attention_type = 'Luong'
beam_width = 3
max_to_keep = 20

epochs = 1000
batch_size = 50

use_attention = False
use_beamsearch = False


# In[ ]:


from hw2_utils_special import MSVD

# dataset_path = 'data_msvd'
dataset_path = sys.argv[1]
msvd = MSVD(dataset_path, training_max_time_steps, word_encoding_threshold)
y_vocab_size = len(msvd.sentenceEncoder.word2int)

if 'train' in mode:
    msvd.load_training_data()
elif 'predict' in mode:
    msvd.load_testing_data()


# In[ ]:


# <S2VT> Sequence to Sequence - Video to Text without CNN

import tensorflow as tf
from tensorflow.python.layers.core import Dense


# In[ ]:


# Parameters

x = tf.placeholder(tf.float32, [None, 80, x_embedding_size], name='x')
y = tf.placeholder(tf.int32, [None, training_max_time_steps + 1], name='y')
x_seq_len = tf.placeholder(tf.int32, [None], name='x_seq_len')
y_seq_len = tf.placeholder(tf.int32, [None], name='y_seq_len')
y_max_seq_len = tf.placeholder(tf.int32, [None], name='y_max_seq_len')
model_batch_size = tf.shape(x)[0]

def rnn_cell():
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    if use_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
    if use_residual:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)
    return cell


# In[ ]:


# Input Embedding [ignored]


# In[ ]:


# Encoder

input_projection_layer = tf.layers.dense(
    inputs=x, 
    units=num_units, 
    use_bias=projection_using_bias
)

# encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
    [rnn_cell() for _ in range(num_layers)])

# initial_state = tf.zeros([tf.size(x_seq_len), encoder_cell.state_size])

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    cell=encoder_cell, 
#     inputs=x, 
    inputs=input_projection_layer,
#     initial_state=initial_state, 
    sequence_length=x_seq_len, 
    dtype=tf.float32)


# In[ ]:


## Decoder for training

embedding = tf.Variable(
#     tf.truncated_normal([y_vocab_size, num_units], mean=0.0, stddev=0.1), 
    tf.random_uniform([y_vocab_size, num_units], -0.1, 0.1), 
    dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, y[:, :-1])

output_projection_layer = Dense(
    y_vocab_size, 
    use_bias=projection_using_bias
)

training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs=encoder_inputs_embedded,
    sequence_length=y_max_seq_len, 
    # although we don't want to feed <eos> into the decoder, still setting seq_len to be max here
    # later it will be filtered out by masks in the loss calculating state
    time_major=False)

# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
    [rnn_cell() for _ in range(num_layers)])

if use_attention:
    ## Attention model
    if attention_type == 'Bahdanau':
        training_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=num_units, 
            memory=encoder_outputs, 
            memory_sequence_length=x_seq_len)
    else:
        training_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=num_units, 
            memory=encoder_outputs, 
            memory_sequence_length=x_seq_len)

    training_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell=decoder_cell, 
        attention_mechanism=training_attention_mechanism, 
        attention_layer_size=num_units)

    training_attention_state = training_attention_cell.zero_state(
        model_batch_size, tf.float32).clone(cell_state=encoder_state)

if not use_attention:
    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell, 
        initial_state=encoder_state,
        helper=training_helper, 
        output_layer=output_projection_layer)
else:
    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=training_attention_cell, 
        initial_state=training_attention_state, 
        helper=training_helper, 
        output_layer=output_projection_layer)

training_maximum_iterations = tf.round(tf.reduce_max(training_max_time_steps))

training_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder=training_decoder, 
    maximum_iterations=training_maximum_iterations)

# epsilon = tf.constant(value=1e-50, shape=[1])
# logits = tf.add(outputs.rnn_output, epsilon)
training_logits = training_outputs.rnn_output
training_id = training_outputs.sample_id


# In[ ]:


# Decoder for predicting
tags = msvd.get_tags()
tag_bos, tag_eos = tags['<bos>'], tags['<eos>']
tag_boses = tf.fill([model_batch_size], tag_bos)

if not use_beamsearch:
    predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=embedding, 
        start_tokens=tag_boses, 
        end_token=tag_eos)

if use_beamsearch:
    # Beam Search tile
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
    tiled_x_seq_len = tf.contrib.seq2seq.tile_batch(x_seq_len, multiplier=beam_width)
    tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)

if use_attention:
    if not use_beamsearch:
        ## Attention model
        if attention_type == 'Bahdanau':
            predicting_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, 
                memory=encoder_outputs, 
                memory_sequence_length=x_seq_len)
        else:
            predicting_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=num_units, 
                memory=encoder_outputs, 
                memory_sequence_length=x_seq_len)
    else:
        ## Attention model
        if attention_type == 'Bahdanau':
            predicting_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, 
                memory=tiled_encoder_outputs, 
                memory_sequence_length=tiled_x_seq_len)
        else:
            predicting_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=num_units, 
                memory=tiled_encoder_outputs, 
                memory_sequence_length=tiled_x_seq_len)

if use_beamsearch:
    if not use_attention:
        beam_initial_state = tiled_encoder_state
#         beam_initial_state = tf.nn.rnn_cell.LSTMStateTuple(
#             tf.contrib.seq2seq.tile_batch(encoder_state[0], multiplier=beam_width),
#             tf.contrib.seq2seq.tile_batch(encoder_state[1], multiplier=beam_width))
    else:
        beam_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=decoder_cell, 
            attention_mechanism=predicting_attention_mechanism, 
            attention_layer_size=num_units)

        # Beam Search decoder
        beam_initial_state = beam_cell.zero_state(
            model_batch_size * beam_width, tf.float32).clone(cell_state=tiled_encoder_state)


if not use_beamsearch:
    if not use_attention:
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell, 
            initial_state=encoder_state, 
            helper=predicting_helper, 
            output_layer=output_projection_layer)
    else:
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=training_attention_cell, 
            initial_state=training_attention_state, 
            helper=predicting_helper, 
            output_layer=output_projection_layer)
else:
    if not use_attention:
        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, 
                initial_state=beam_initial_state, 
                embedding=embedding,
                start_tokens=tag_boses, 
                end_token=tag_eos, 
                beam_width=beam_width,
                output_layer=output_projection_layer,
                length_penalty_weight=0.0)
    else:
        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=beam_cell,
                initial_state=beam_initial_state, 
                embedding=embedding,
                start_tokens=tag_boses, 
                end_token=tag_eos, 
                beam_width=beam_width,
                output_layer=output_projection_layer,
                length_penalty_weight=0.0)

predicting_maximum_iterations = tf.round(tf.reduce_max(training_max_time_steps) * 2)

predicting_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder=predicting_decoder, 
    impute_finished=False, 
    maximum_iterations=predicting_maximum_iterations)

if not use_beamsearch:
    predicting_id = predicting_outputs.sample_id
else:
    predicting_id = predicting_outputs.predicted_ids[:, :, 0]


# In[ ]:


# Loss and Optimizer

targets = y[:, 1:]

masks = tf.sequence_mask(y_seq_len, training_max_time_steps, dtype=tf.float32)
loss = tf.contrib.seq2seq.sequence_loss(
    logits=training_logits, 
    targets=targets, 
    weights=masks, 
    average_across_timesteps=False, 
    average_across_batch=True)
loss = tf.reduce_sum(loss)

# params = tf.trainable_variables()
# gradients = tf.gradients(loss, params)
# clipped_gradients, _ = tf.clip_by_global_norm(
#     gradients, max_gradient_norm)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# max_gradient_norm = 1.0
# learning_rate = 0.001
# max_gradient_norm = 10.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# update_step = optimizer.apply_gradients(
#     zip(clipped_gradients, params))


# In[ ]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# In[ ]:


import numpy as np
from hw2_utils_special import Predictions

saver = tf.train.Saver(max_to_keep=max_to_keep)
predictions = Predictions(msvd)

if 'train' in mode:
    if 'load' in mode:
        saver.restore(sess, '{}.ckpt'.format(model_name))
    
    best_loss_val = 1e10
    
    if not random_every_epoch:
        msvd.set_captions_randomly()

    for epoch in range(epochs):
        if random_every_epoch:
            msvd.set_captions_randomly()
        count = 0
        loss_val_sum = 0.0
        for x_, y_, x_seq_len_, y_seq_len_ in msvd.next_batch(batch_size):
            if shuffle_training_data:
                p = np.random.permutation(x_.shape[0])
                x_, y_, x_seq_len_, y_seq_len_ = x_[p], y_[p], x_seq_len_[p], y_seq_len_[p]
            _, loss_val, preds = sess.run(
                [train_op, loss, training_id], 
                feed_dict={x: x_, 
                           y: y_, 
                           x_seq_len: x_seq_len_,
                           y_seq_len: y_seq_len_, 
                           y_max_seq_len: np.full(y_seq_len_.size, 
                                                  training_max_time_steps, 
                                                  dtype=np.int32)})
            count += 1
            loss_val_sum += loss_val
#             print('Epoch {}, train_loss_val: {}.'.format(epoch, loss_val))
#             predictions.print(preds, False, True, '=> {}')
            if loss_val < best_loss_val:
                best_loss_val = loss_val
#                 print('Model saved.')
#                 saver.save(sess, '{}.ckpt'.format(model_name))
            if save_per_epoch and (epoch+1) % save_per_epoch == 0:
                print('Model saved.')
                saver.save(sess, '{}_epoch_{}.ckpt'.format(model_name, epoch))
        loss_val_avg = loss_val_sum / count
        predictions.print(preds[:5], False, True, '=> {}')
        print('Epoch {}, average_train_loss_val: {}.'.format(epoch, loss_val_avg))

    print('Finished training. Best train_loss_val: {}.'.format(best_loss_val))
    
elif 'predict' in mode:
    if iter_epochs and 'load' in mode:
        for epoch in iter_epochs:
            saver.restore(sess, '{}_epoch_{}.ckpt'.format(model_name, epoch))
            for x_, x_seq_len_, x_id in msvd.testing_data(batch_size):
                preds = sess.run(
                    predicting_id, 
                    feed_dict={x: x_, 
                               x_seq_len: x_seq_len_})
                predictions.print(preds, False, True, '=> {}')
                predictions.add(x_id, preds)

            predictions.save('{}_epoch_{}.txt'.format(prediction_name, epoch))

        print('Finished predicting.')
    else:
        if 'load' in mode:
            saver.restore(sess, '{}.ckpt'.format(model_name))

        for x_, x_seq_len_, x_id in msvd.testing_data(batch_size):
            preds = sess.run(
                predicting_id, 
                feed_dict={x: x_, 
                           x_seq_len: x_seq_len_})
            predictions.print(preds, False, True, '=> {}')
            predictions.add(x_id, preds)

        predictions.save(prediction_name + '_tmp')
#         predictions.save('{}.txt'.format(prediction_name))

        print('Finished predicting.')


# In[ ]:


from hw2_eval_special import special_mission
special_mission(prediction_name + '_tmp', prediction_name, True)
from os import remove
remove(prediction_name + '_tmp')

