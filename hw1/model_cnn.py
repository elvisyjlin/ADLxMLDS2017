import sys
from mypackages.utils import *
from mypackages.models import train_cnn

load_timit_dataset(dataset_path=sys.argv[1])
X_train, y_train, _, _, _ = preprocess()
train_cnn(X_train, y_train, epochs=7, conv_activation='linear', model_name='cnn.mdl')
