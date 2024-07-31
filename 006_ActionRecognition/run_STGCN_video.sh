#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

python3 app.py -i raspipose/mmskeleton/resource/data_example/clean_and_jerk.mp4 -t yolov8 -act
