import itertools
import numpy as np
import pandas as pd
from keras.models import load_model
from os.path import join
from sklearn.preprocessing import LabelBinarizer

def load_timit_dataset(flag='mfcc', dataset_path='data'):
	print('# Loading dataset...')

	global mfcc_table, fbank_table, mfcc_test, fbank_test
	global labels_train, map_48phone_char, map_48_39

	## declares all paths
	mfcc_train_path = join(dataset_path, 'mfcc/train.ark')
	mfcc_test_path = join(dataset_path, 'mfcc/test.ark')
	fbank_train_path = join(dataset_path, 'fbank/train.ark')
	fbank_test_path = join(dataset_path, 'fbank/test.ark')
	labels_train_path = join(dataset_path, 'label/train.lab')
	map_48phone_char_path = join(dataset_path, '48phone_char.map')
	map_48_39_path = join(dataset_path, 'phones/48_39.map')
	
	## reads data from files
	if flag == 'mfcc':
		mfcc_train = pd.read_csv(mfcc_train_path, sep=' ', header=None)
		mfcc_test = pd.read_csv(mfcc_test_path, sep=' ', header=None)
	else:
		fbank_train = pd.read_csv(fbank_train_path, sep=' ', header=None)
		fbank_test = pd.read_csv(fbank_test_path, sep=' ', header=None)
		
	labels_train = pd.read_csv(labels_train_path, sep=',', header=None)
	map_48phone_char = pd.read_csv(map_48phone_char_path, sep='\t', header=None)
	map_48_39 = pd.read_csv(map_48_39_path, sep='\t', header=None)

	## sorts data and labels by their IDs(person_sentence_frame)
	if flag == 'mfcc':
		mfcc_table = pd.merge(mfcc_train, labels_train, on=0)
	else:
		fbank_table = pd.merge(fbank_train, labels_train, on=0)

def preprocess(flag='mfcc'):
	print('# Preprocessing...')

	global X_train_mfcc, X_train_fbank, y_train_mfcc, y_train_fbank
	global X_test_mfcc, id_test_mfcc, X_test_fbank, id_test_fbank
	global label_binarizer_mfcc, label_binarizer_fbank
	
	## converts pandas dataframe into numpy ndarray
	if flag == 'mfcc':
		X_train_mfcc = mfcc_table.iloc[:, 1:-1].as_matrix()
		X_test_mfcc = mfcc_test.iloc[:, 1:].as_matrix()
		id_test_mfcc = mfcc_test.iloc[:, 0]
		y_train_mfcc = mfcc_table.iloc[:, -1]
	else:
		X_train_fbank = fbank_table.iloc[:, 1:-1].as_matrix()
		X_test_fbank = fbank_test.iloc[:, 1:].as_matrix()
		id_test_fbank = fbank_test.iloc[:, 0]
		y_train_fbank = fbank_table.iloc[:, -1]
	## converts pandas dataframe into dictionary
	map_48_39_dict = map_48_39.set_index(0).to_dict()[1]

	if flag == 'mfcc':
		## maps 48 phones onto 39 phones
		y_train_mfcc = np.vectorize(map_48_39_dict.get)(y_train_mfcc)
		## turn labels into one hot representation
		label_binarizer_mfcc = LabelBinarizer()
		label_binarizer_mfcc.fit(y_train_mfcc)
		y_train_mfcc = label_binarizer_mfcc.transform(y_train_mfcc)
	else:
		# ## maps 48 phones onto 39 phones
		y_train_fbank = np.vectorize(map_48_39_dict.get)(y_train_fbank)
		# ## turn labels into one hot representation
		label_binarizer_fbank = LabelBinarizer()
		label_binarizer_fbank.fit(y_train_fbank)
		y_train_fbank = label_binarizer_fbank.transform(y_train_fbank)

	if flag == 'mfcc':
		return X_train_mfcc, y_train_mfcc, X_test_mfcc, id_test_mfcc, label_binarizer_mfcc
	else:
		return X_train_fbank, y_train_fbank, X_test_fbank, id_test_fbank, label_binarizer_fbank

def pad_and_reshape(X_train, y_train, n_timesteps):
	## reshapes data and labels according to n_timesteps
	if X_train.shape[0] % n_timesteps != 0:
		X_train = np.lib.pad(X_train, 
							 ((0,n_timesteps-X_train.shape[0]%n_timesteps),(0,0)), 
							 mode='wrap')
	X_train = X_train.reshape((X_train.shape[0]//n_timesteps, 
							   n_timesteps,X_train.shape[1]))
	if y_train.shape[0] % n_timesteps != 0:
		y_train = np.lib.pad(y_train, 
							 ((0,n_timesteps-y_train.shape[0]%n_timesteps), (0,0)), 
							 mode='wrap')
	y_train = y_train.reshape((y_train.shape[0]//n_timesteps, 
							   n_timesteps,y_train.shape[1]))
	return X_train, y_train

def predict(model_name, X_test, id_test, label_binarizer, 
			n_timesteps=123, threshold=0.7, output_file='default.csv'):
	print('# Predicting...')

	## reshape the data to fit the model
	if X_test.shape[0] % n_timesteps != 0:
		X_test = np.lib.pad(X_test, 
							 ((0,n_timesteps-X_test.shape[0]%n_timesteps),(0,0)), 
							 mode='wrap')
	X_test = X_test.reshape((X_test.shape[0]//n_timesteps, 
							   n_timesteps,X_test.shape[1]))

	## loads the model
	model_path = '../models/' + model_name
	model = load_model(model_path)

	## prints information of the model
	print('Summary')
	print(model.summary())
	print('Metric Names')
	print(model.metrics_names)

	## makes predictions
	y_predict = model.predict(X_test)
	
	## postprocesses the predictions to csv format
	postprocess(y_predict, id_test, label_binarizer, model_name, threshold, output_file)

## Postprocess the results of predictions
## if you want to disable threshold, set threshold=None
def postprocess(y_predict, id_test, label_binarizer, model_name, threshold, output_file):
	print('# Postprocessing...')

	y_predict = y_predict.reshape((y_predict.shape[0]*y_predict.shape[1], y_predict.shape[2]))
	y_predict = y_predict[:id_test.shape[0], :]
	y_predict_max = y_predict.max(axis=1)

	indices = y_predict.argmax(axis=1)
	y_predict[np.arange(indices.size), indices] = 1

	y_predict = label_binarizer.inverse_transform(y_predict)

	map_48phone_char_dict = map_48phone_char.set_index(0).to_dict()[2]
	y_predict = np.vectorize(map_48phone_char_dict.get)(y_predict)

	predict_table = pd.DataFrame({0: id_test, 1: y_predict, 2: y_predict_max})
	if threshold:
		predict_table = predict_table[predict_table[2] >= threshold]
	predict_table[0] = predict_table[0].map(lambda x: '_'.join(x.split('_')[:-1]))
	predict_table = predict_table.groupby(0)[1].apply(''.join).reset_index()

	output_csv = predict_table.rename(columns={0: 'id', 1: 'phone_sequence'})
	
	output_csv['phone_sequence'] = output_csv['phone_sequence'].map(lambda x: ''.join(i for i, _ in itertools.groupby(x.strip('L'))))
	output_csv.to_csv(output_file, sep=',', header=True, index=False)
