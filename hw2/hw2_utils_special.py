import numpy as np
import string
from collections import defaultdict

class SentenceEncoder():
    def __init__(self):
        self.defaultTags = ['<pad>', '<bos>', '<eos>', '<unk>']
        
        self.stop_words = string.punctuation + '“' + '”'
        self.trantab = str.maketrans('', '', self.stop_words)
        
        self.word2int = None
        self.int2word = None
        self.bias = None
        
        self.ready = False
        
    def fit(self, sentences, threshold=1):
        word_count = defaultdict(int)
        num_sentense = 0
        for sentence in sentences:
            num_sentense += 1
            for word in self.split(sentence):
                word_count[word] += 1
        
        vocab = [word for word in word_count if word_count[word] >= threshold]
        print('Filtered words from {} to {}.'.format(len(word_count), len(vocab)))
        
        self.word2int = {}
        self.int2word = {}
        for i, w in enumerate(self.defaultTags):
            self.word2int[w] = i
            self.int2word[i] = w
            word_count[w] = num_sentense
        for i, w in enumerate(sorted(vocab)):
            self.word2int[w] = i + 4
            self.int2word[i + 4] = w
        
        self.bias = np.array([1.0 * word_count[self.int2word[i]]] for i in self.int2word)
                
    def split(self, sentence):
        return sentence.translate(self.trantab).lower().split()
    
    def lookup(self, word):
        return self.word2int[word] if word in self.word2int else self.word2int['<unk>']
    
    def transform(self, sentence):
        return np.array([self.word2int['<bos>']] + 
                        [self.lookup(word) for word in self.split(sentence)] + 
                        [self.word2int['<eos>']])
        
    def inverse_transform(self, ints):
        tag_eos = self.lookup('<eos>')
        if tag_eos in ints:
            ints = ints[:np.argmax(np.array(ints) == tag_eos) + 1]
        sentence = ' '.join([self.int2word[int] for int in ints])
        sentence = sentence.replace('<bos> ', '').replace(' <eos>', '')
        return sentence

    def getBiasVector(self):
        return self.bias

import json
import numpy as np
from os import listdir
from os.path import join
from pprint import pprint
from random import randrange

class MSVD():
    def __init__(self, path, training_max_time_steps=40, word_encoding_threshold=None):
        self.path = path
        self.training_max_time_steps = 1
        if training_max_time_steps > 0:
            self.training_max_time_steps = training_max_time_steps
        self.id_train = []
        self.id_test = []
        self.label_dict = {}
        self.x_train = np.zeros((1450, 80, 4096), dtype=np.float32)
        self.x_test = np.zeros((100, 80, 4096), dtype=np.float32)
        self.y_train = np.zeros((1450, self.training_max_time_steps + 1), dtype=np.int32) # y_train is longer than its length by 1
        self.x_seq_len = np.zeros((1450), dtype=np.int32)
        self.x_test_seq_len = np.zeros((100), dtype=np.int32)
        self.y_seq_len = np.zeros((1450), dtype=np.int32)
        self.sentenceEncoder = SentenceEncoder()
        self.ready = False
        self.train_loaded = False
        self.test_loaded = False
        
        label_train_path = join(self.path, 'training_label.json')
        with open(label_train_path, 'r', encoding='utf-8') as f:
            label_train = json.load(f)
        print('Loaded MSVD labels.')
        
        for label in label_train:
            self.label_dict[label['id']] = label['caption']
        if word_encoding_threshold:
            self.sentenceEncoder.fit(self.get_captions(), word_encoding_threshold)
        else:
            self.sentenceEncoder.fit(self.get_captions())
        
    def load_training_data(self):
        if self.train_loaded: return
        
        feature_train_path = join(self.path, 'training_data/feat')
            
        index = 0
        for file in listdir(feature_train_path):
            id = '.'.join(file.split('.')[:-1])
            path = join(feature_train_path, file)

            self.id_train.append(id)
            self.x_train[index] = np.load(path)
            self.x_seq_len[index] = self.x_train[index].shape[0]
            index += 1
        
        self.train_loaded = True
        print('Loaded MSVD training dataset.')
       
    def load_testing_data(self):
        if self.test_loaded: return
        
        print('MSVD load_testing_data()')
        feature_test_path = join(self.path, 'testing_data/feat')
            
        index = 0
        for file in listdir(feature_test_path):
            id = '.'.join(file.split('.')[:-1])
            path = join(feature_test_path, file)

            self.id_test.append(id)
            self.x_test[index] = np.load(path)
            self.x_test_seq_len[index] = self.x_test[index].shape[0]
            index += 1
            
        self.test_loaded = True
        print('Loaded MSVD testing dataset.')
    
    def set_captions_by_default(self):
        for index in range(len(self.id_train)):
            id = self.id_train[index]
            choice = 0
            label = self.sentenceEncoder.transform(self.label_dict[id][choice])
            label_len = len(label) - 1
            if label_len > self.training_max_time_steps:
                self.y_train[index] = np.concatenate((label[:self.training_max_time_steps-1+1], [label[-1]]), axis=0)
                self.y_seq_len[index] = self.training_max_time_steps
            elif label_len < self.training_max_time_steps:
                self.y_train[index] = np.pad(label, 
                                             (0, self.training_max_time_steps-label.shape[0]+1), 
                                             'constant', 
                                             constant_values=0)
                self.y_seq_len[index] = label_len
            else:
                self.y_train[index] = label
                self.y_seq_len[index] = label_len
        self.ready = True
        
    def set_captions_randomly(self):
        for index in range(len(self.id_train)):
            id = self.id_train[index]
            choice = randrange(0, len(self.label_dict[id]))
            label = self.sentenceEncoder.transform(self.label_dict[id][choice])
            label_len = len(label) - 1
            if label_len > self.training_max_time_steps:
                self.y_train[index] = np.concatenate((label[:self.training_max_time_steps-1+1], [label[-1]]), axis=0)
                self.y_seq_len[index] = self.training_max_time_steps
            elif label_len < self.training_max_time_steps:
                self.y_train[index] = np.pad(label, 
                                             (0, self.training_max_time_steps-label.shape[0]+1), 
                                             'constant', 
                                             constant_values=0)
                self.y_seq_len[index] = label_len
            else:
                self.y_train[index] = label
                self.y_seq_len[index] = label_len
        self.ready = True
    
    def next_batch(self, batch_size):
        if not self.ready:
            raise Exception('MSVD is not ready. Set captions before getting the batches!')
        
        for idx in range(0, 1450, batch_size):
            yield [self.x_train[idx:idx+batch_size], 
                   self.y_train[idx:idx+batch_size],
                   self.x_seq_len[idx:idx+batch_size],
                   self.y_seq_len[idx:idx+batch_size]]
    
    def testing_data(self, batch_size):
        for idx in range(0, 100, batch_size):
            yield [self.x_test[idx:idx+batch_size], 
                   self.x_test_seq_len[idx:idx+batch_size],
                   self.id_test[idx:idx+batch_size]]
    
    def get_captions(self):
        return sum(self.label_dict.values(), [])
    
    def get_tags(self):
        return dict((w, i) for i, w in enumerate(self.sentenceEncoder.defaultTags))

from os.path import join

class Predictions():
    def __init__(self, msvd, path='.'):
        self.msvd = msvd
        self.path = path
        self.predictions = {}
        
    def add(self, ids, preds):
        for id, pred in zip(ids, preds):
            pred = self.msvd.sentenceEncoder.inverse_transform(pred)
            self.predictions[id] = pred
    
    def print(self, preds, numpy=True, sentence=True, formatted='{}'):
        for pred in preds:
            if numpy: print(formatted.format(pred))
            if sentence: print(formatted.format(self.msvd.sentenceEncoder.inverse_transform(pred)))
            
    def save(self, filename):
        with open(filename, 'w') as f:
            for id, pred in self.predictions.items():
                f.write('{},{}\n'.format(id, pred))
        print('Saved predictions as {}.'.format(filename))
