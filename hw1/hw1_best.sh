#!/bin/bash

wget -O models/bset.mdl "https://www.dropbox.com/s/g082zogqk15rye3/best.mdl?dl=1"
python predict_best.py "$1" "$2"
