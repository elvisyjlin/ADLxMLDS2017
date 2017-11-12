#!/bin/bash

wget -O "models/hw2_special.ckpt.data-00000-of-00001" "https://www.dropbox.com/s/6j9oy40cs8dlcj7/hw2_special.ckpt.data-00000-of-00001?dl=1"
python hw2_special.py "$1" "$2"
