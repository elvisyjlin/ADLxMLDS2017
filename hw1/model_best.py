import sys
from mypackages.utils import *
from mypackages.models import train_rnn

load_timit_dataset(dataset_path=sys.argv[1])
X_train, y_train, _, _, _ = preprocess()
train_best(X_train, y_train, epochs=13, model_name='best.mdl')
