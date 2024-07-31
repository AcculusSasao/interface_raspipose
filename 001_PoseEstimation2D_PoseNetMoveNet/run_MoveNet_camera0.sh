#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

python3 app.py -i 0 -t movenet
