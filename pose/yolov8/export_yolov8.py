import sys
sys.path.insert(0, 'ultralytics_yolov8')
from ultralytics import YOLO

imgsz = 320
if len(sys.argv) > 1:
    imgsz = int(sys.argv[1])

model = YOLO('yolov8n-pose')
model.export(
    format='tflite',
    imgsz=imgsz,
    data="coco128.yaml",
    int8=True,
    separate_outputs=True,
    export_hw_optimized=True,
    simplify=True,
    uint8_io_dtype=False,
)
