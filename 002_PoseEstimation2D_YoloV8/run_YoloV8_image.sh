#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

python3 app.py -i testimg/1.jpg -t yolov8
