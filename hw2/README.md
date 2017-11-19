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
```
