#!/bin/bash

wget -O models/best.mdl "https://www.dropbox.com/s/g082zogqk15rye3/best.mdl?dl=1"
python mypackages/predict_best.py "$1" "$2"
