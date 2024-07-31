#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

python3 app.py -i testimg/3.jpg -t mediapipe -angle -3
