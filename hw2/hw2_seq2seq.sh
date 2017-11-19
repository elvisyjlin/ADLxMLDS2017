#!/bin/bash

# TODO!!!!!
wget -O "[model name]" "[model url]"
python hw2_seq2seq.py "$1" "[model]" "$2"
python hw2_seq2seq.py "$1/peer_review" "[model]"  "$3"
