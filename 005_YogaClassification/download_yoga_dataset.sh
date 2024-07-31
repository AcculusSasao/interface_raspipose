#!/bin/sh

cd $(dirname $(readlink -f $0))/../pose

wget http://download.tensorflow.org/data/pose_classification/yoga_poses.zip -O yoga_poses.zip
mkdir -p yoga_poses && cd yoga_poses
unzip -qo ../yoga_poses.zip
rm -f ../yoga_poses.zip
