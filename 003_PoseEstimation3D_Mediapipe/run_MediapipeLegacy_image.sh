#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

python3 app.py -i testimg/2.jpg -t mediapipe_legacy -3
