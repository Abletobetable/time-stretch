#!/bin/bash
pip install -q librosa soundfile
python3 main.py \
--input $1 \
--output $2 \
--ratio $3