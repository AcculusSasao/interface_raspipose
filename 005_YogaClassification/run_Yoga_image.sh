#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

python3 app.py -i yoga_poses/test/dog/guy3_dog114.jpg -t movenet -y
