## HW 4 - Generative Adversarial Networks

### Information

Code Deadline (GitHub): 12/31/2017 23:59; postponed to 01/07/2018 23:59

### Instructions

<!-- under/hw4: run.sh, train.py, (pre-)trained_model, generate.py, samples/, report.pdf -->
<!-- bash run.sh [testing_text.txt] -->
<!-- sample_(testing_text_id)_(sample_id).jpg ; five images for each text -->

To generate amine faces according to texts:
```bash
bash run.sh [testing_text.txt] # will save samples to sample/
```

To train a text2image generative adversarial networks:
```python
train.py [dataset_path] [model_name] # will save model to [model_name].D.h5 and [model_name].G.h5
```
Note that train.py is designed for anime faces dataset training. 
I use the preprocessed onehot_labels.npy as its labels.

To train a CycleGAN:
```python
train_bonus.py [output_path]
```
Note that you need to prepare datasets of two domain first and set their paths in (train_A, train_B) in the code.

### Used Packages

In Python 3.6.

1. TensorFlow 1.3.0
2. Keras 2.0.7
3. Numpy 1.13.3

### About Models

#### Text2image GAN

```python
```

#### CycleGAN

```python
nc_in = 3
nc_out = 3
ngf = 64
ndf = 64
use_lsgan = True
λ = 10 if use_lsgan else 100

loadSize = 143
imageSize = 128
batchSize = 4
lrD = 2e-4
lrG = 2e-4
```