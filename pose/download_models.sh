#!/bin/sh

OUTDIR=models
mkdir -p $OUTDIR

# google posenet
wget -N -P $OUTDIR https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
wget -N -P $OUTDIR https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_513x513_multi_kpt_stripped.tflite

# google movenet
curl -L -o $OUTDIR/singlepose-lightning-tflite-int8.tar.gz \
	https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-lightning-tflite-int8/1/download
tar xf $OUTDIR/singlepose-lightning-tflite-int8.tar.gz
mv 4.tflite $OUTDIR/singlepose-lightning-tflite-int8.tflite
rm -f $OUTDIR/singlepose-lightning-tflite-int8.tar.gz

curl -L -o $OUTDIR/singlepose-thunder-tflite-int8.tar.gz \
	https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-thunder-tflite-int8/1/download
tar xf $OUTDIR/singlepose-thunder-tflite-int8.tar.gz
mv 4.tflite $OUTDIR/singlepose-thunder-tflite-int8.tflite
rm -f $OUTDIR/singlepose-thunder-tflite-int8.tar.gz

# google yoga pose classifier
wget -N -P $OUTDIR https://storage.googleapis.com/download.tensorflow.org/models/tflite/pose_classifier/yoga_classifier.tflite

# mediapipe pose_landmarker
wget -N -P $OUTDIR https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
wget -N -P $OUTDIR https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
wget -N -P $OUTDIR https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
