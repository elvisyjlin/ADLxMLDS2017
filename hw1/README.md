## HW 1 - Sequence Labeling

# Information

Competition Period (Kaggle): 10/05/2017 12:00 - 10/28/2017 12:00

Code Deadline (GitHub): 10/28/2017 23:59


# Instructions

To predict labels of the testing data:
```bash
bash hw1_rnn.sh [input path] [output path]
bash hw1_cnn.sh [input path] [output path]
bash hw1_best.sh [input path] [output path]
```


To train models from the training data:
```bash
python model_rnn.py [input path]
python model_cnn.py [input path]
python model_best.py [input path]
```


Run test.sh to do all the procedures mentioned above:
```bash
bash test.sh [input path]
```
The output paths are rnn.mdl, cmm.mdl and best.mdl under hw1/models.
