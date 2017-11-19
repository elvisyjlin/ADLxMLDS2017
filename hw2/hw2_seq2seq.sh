#!/bin/bash

wget -O "models/hw2_as2vt.ckpt.data-00000-of-00001" "https://www.dropbox.com/s/1c5bp6khol8li04/hw2_as2vt.data-00000-of-00001?dl=1"
python hw2_seq2seq.py "$1" "hw2_as2vt" "$2"
python hw2_seq2seq.py "$1/peer_review" "hw2_as2vt"  "$3"
