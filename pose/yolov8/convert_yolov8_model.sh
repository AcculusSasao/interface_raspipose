#!/bin/sh

pip3 install -r requirements_export.txt

# original export
#pip install ultralytics
#yolo export model='yolov8n-pose.pt' format='tflite' int8=True

git clone --recursive https://github.com/DeGirum/ultralytics_yolov8.git
cd ultralytics_yolov8
git checkout 123c024b9e353828196b072c2288c1faf77f9b72
cd -
python3 export_yolov8.py 320

mkdir -p ../models
cp yolov8n-pose_saved_model/yolov8n-pose_full_integer_quant.tflite \
	../models/yolov8n-pose_full_integer_quant_320.tflite
