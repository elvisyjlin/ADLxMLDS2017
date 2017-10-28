from mypackages.utils import *
from mypackages.models import train_rnn

load_timit_dataset()
X_train, y_train, _, _, _ = preprocess()
train_rnn(X_train, y_train, epochs=7, model_name='models/rnn.mdl')
