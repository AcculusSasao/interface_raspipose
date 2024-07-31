#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

python3 app.py -i testimg/4.jpg -t yolov8 -act --use_image_as_video
