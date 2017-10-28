import sys
from mypackages.utils import predict

load_timit_dataset(sys.argv[1])
_, _, X_test, id_test, label_binarizer = preprocess()
predict(model_name='models/cnn.mdl', X_test=X_test, id_test=id_test, 
		label_binarizer=label_binarizer, threshold=0.7)
