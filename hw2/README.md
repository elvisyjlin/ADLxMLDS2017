## HW 2 - Video Captioning

### Information

Special Mission Deadline (Github): 11/12/2017 23:59  
Need to submit 5 captions of assigned testing videos to the Google form.

Code Deadline (GitHub): 11/19/2017 23:59

### Instructions

#### Special Mission

To predict captions of the testing data:
```bash
bash hw2_special.sh [the data directory] [output file]
```

### Final Submission

To predict captions of the testing data with the best model:
```bash
bash hw2_seq2seq.sh [the data directory] [output file of testing data] [output file of peer review]
```

Or with a custom model:
```bash
bash hw2_seq2seq.py [the data directory] [model_file] [output file]
```

To train a model from MSVD dataset:
```bash
bash model_seq2seq.py [the data directory] [model file]
```

### Used Packages

1. TensorFlow 1.3.0
2. Numpy 1.13.3

### About Models

#### Special Mission

Trained for 600 epochs.  
Using the simple seq2seq model with the following parameters:

```python
training_max_time_steps = 40
word_encoding_threshold = 1
random_every_epoch = True
shuffle_training_data = True

num_units = 256
num_layers = 2

use_dropout = True
output_keep_prob = 0.5
use_residual = True 
projection_using_bias = False
beam_width = 3

epochs = 1000
batch_size = 50
use_attention = False
use_beamsearch = False
```

#### Final Submission

```python
y_max_length = 20
word_encoding_threshold = 1

num_units = 256
epochs = 420
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
sampling_decaying_rate = None # a float number between 0~1 e.g. 0.99
sampling_decaying_mode = None # 'linear or 'exponential'
sampling_decaying_per_epoch = None # an integer number
use_embedding = 'fasttext' # None, 'word2vec', 'glove', or 'fasttext'
```
