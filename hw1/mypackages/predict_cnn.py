import sys
from utils import *

load_timit_dataset(dataset_path=sys.argv[1])
_, _, X_test, id_test, label_binarizer = preprocess()
predict(model_name='cnn.mdl', X_test=X_test, id_test=id_test, 
		label_binarizer=label_binarizer, threshold=0.7, output_file=sys.argv[2])
