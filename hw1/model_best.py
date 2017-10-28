from mypackages.utils import *
from mypackages.models import train_rnn

load_timit_dataset()
X_train, y_train, _, _, _ = preprocess()
train_best(X_train, y_train, epochs=13, model_name='best.mdl')
