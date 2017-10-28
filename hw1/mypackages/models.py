from os.path import join
from keras.models import Sequential
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import LSTM, Dense, Conv2D, Activation, Dropout, Reshape
from sklearn.model_selection import train_test_split
from utils import pad_and_reshape

## train the RNN model
def train_rnn(X_train, y_train, n_timesteps=123, 
			  epochs=20, batch_size=32, 
			  validating_ratio=0.1, lstm_hidden_nums=512, 
			  nb_classes=39, model_name='default'):
	print('# Training the RNN model...')
	
	## reshapes data and labels according to n_timesteps
	X_train, y_train = pad_and_reshape(X_train, y_train, n_timesteps)

	## splits inputs into training and validating inputs
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
														  y_train, 
														  test_size=validating_ratio, 
														  random_state=0)
	
	## contructs the model
	model = Sequential()
	model.add( Bidirectional( LSTM(lstm_hidden_nums, return_sequences=True), 
							 input_shape=(n_timesteps, X_train.shape[2]) ) )
	model.add( TimeDistributed( Dense(nb_classes) ) )
	model.add( Activation('softmax') )
	model.compile(loss='categorical_crossentropy', 
				  optimizer='rmsprop', 
				  metrics=['accuracy'])

	## prints information of the model
	print('Summary')
	print(model.summary())
	print('Metric Names')
	print(model.metrics_names)

	## train the model
	model.fit(X_train, y_train, 
			  epochs=epochs,  
			  batch_size=batch_size, 
			  shuffle=True, 
			  validation_data=(X_valid, y_valid))

	## saves the model
	model_path = join('models', model_name)
	model.save(model_path)

## train the CNN model
def train_cnn(X_train, y_train, n_timesteps=123, 
			  epochs=20, batch_size=32, 
			  conv_hidden_size=256, conv_size=(3, 3), conv_activation='linear', 
			  validating_ratio=0.1, lstm_hidden_nums=512, 
			  nb_classes=39, model_name='default'):
	print('# Training the CNN model...')
	
	## reshapes data and labels according to n_timesteps
	X_train, y_train = pad_and_reshape(X_train, y_train, n_timesteps)

	## splits inputs into training and validating inputs
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
														  y_train, 
														  test_size=validating_ratio, 
														  random_state=0)
	
	## contructs the model
	model = Sequential()
	model.add( Reshape((X_train.shape[1], X_train.shape[2], 1), input_shape=(X_train.shape[1], X_train.shape[2])) )
	model.add( Conv2D(conv_hidden_size, conv_size, padding='same') )
	model.add( Activation(conv_activation) )
	model.add( Reshape((X_train.shape[1], X_train.shape[2] * conv_hidden_size)) )
	model.add( TimeDistributed( Dense(X_train.shape[2]) ) )
	model.add( Bidirectional( LSTM(lstm_hidden_nums, return_sequences=True)) )
	model.add( TimeDistributed( Dense(nb_classes) ) )
	model.add( Activation('softmax') )
	model.compile(loss='categorical_crossentropy', 
				  optimizer='rmsprop', 
				  metrics=['accuracy'])

	## prints information of the model
	print('Summary')
	print(model.summary())
	print('Metric Names')
	print(model.metrics_names)

	## train the model
	model.fit(X_train, y_train, 
			  epochs=epochs,  
			  batch_size=batch_size, 
			  shuffle=True, 
			  validation_data=(X_valid, y_valid))

	## saves the model
	model_path = join('models', model_name)
	model.save(model_path)
	
## train the best model
def train_best(X_train, y_train, n_timesteps=123, 
			   epochs=20, batch_size=32, 
			   lstm_hidden_nums=[512,512,512,512], 
			   nb_classes=39, model_name='default'):
	print('# Training the best model...')
	
	## reshapes data and labels according to n_timesteps
	X_train, y_train = pad_and_reshape(X_train, y_train, n_timesteps)
	
	# trains the model with all available data
	X_valid = X_train
	y_valid = y_train
	
	## contructs the model
	model = Sequential()
	model.add( Bidirectional( LSTM(lstm_hidden_nums[0], return_sequences=True), 
							 input_shape=(n_timesteps, X_train.shape[2]) ) )
	model.add( Bidirectional( LSTM(lstm_hidden_nums[1], return_sequences=True) ) )
	model.add( Dropout(0.5) )
	model.add( Bidirectional( LSTM(lstm_hidden_nums[2], return_sequences=True) ) )
	model.add( Bidirectional( LSTM(lstm_hidden_nums[3], return_sequences=True) ) )
	model.add( Dropout(0.5) )
	model.add( TimeDistributed( Dense(nb_classes, activation='softmax') ) )
	model.compile(loss='categorical_crossentropy', 
				  optimizer='rmsprop', 
				  metrics=['accuracy'])

	## prints information of the model
	print('Summary')
	print(model.summary())
	print('Metric Names')
	print(model.metrics_names)

	## train the model
	model.fit(X_train, y_train, 
			  epochs=epochs,  
			  batch_size=batch_size, 
			  shuffle=True, 
			  validation_data=(X_valid, y_valid))

	## saves the model
	model_path = join('models', model_name)
	model.save(model_path)
