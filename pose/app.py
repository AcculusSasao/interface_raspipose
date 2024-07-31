# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import argparse
import cv2
import numpy as np
import sys
import os
import time

from raspipose import PoseEstimation, PoseEstimationMediapipeLegacyParam, PoseEstimationMediapipeParam
from raspipose import YogaClassifier
from raspipose import PoseStockSingle, ActionRecognitionSTGCN

'''
Pose Estimation
    Posenet : https://www.tensorflow.org/lite/examples/pose_estimation/overview?hl=ja
        output: 17 keypoints, single person
    Movenet : https://www.tensorflow.org/lite/examples/pose_estimation/overview?hl=ja
        output: 17 keypoints, single person
    YoloV8 : https://github.com/ultralytics/ultralytics
        output: 17 keypoints, multi person
    Mediapipe(legacy I/F) : https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
        output: 33 keypoints, single person
    Mediapipe : https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        output: 33 keypoints, multi person
Pose Classification
    Yoga Classifier : https://www.tensorflow.org/lite/tutorials/pose_classification?hl=ja
        input: 17 keypoints
Action Recognition
    mmskeleton (ST-GCN) : https://github.com/open-mmlab/mmskeleton
        input: 18 keypoints
'''

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='mediapipe', choices=['posenet', 'movenet', 'yolov8', 'mediapipe', 'mediapipe_legacy'], help='pose estimation type')
    parser.add_argument('-i', '--input', default=None, help='camera index number, image filename, video filename, or None(random data)')

    parser.add_argument('-y', '--yoga', action='store_true', help='do yoga classification')
    parser.add_argument('-3', '--show_3d', action='store_true', help='show 3D pose (type=mediapipe* only)')
    parser.add_argument('-angle', '--detect_spine_angle', action='store_true', help='show 3D pose (type=mediapipe* only)')
    parser.add_argument('-act', '--action_recognition', action='store_true', help='classify actions by ST-GCN')

    parser.add_argument('-ss', '--show_scale', type=float, default=1.0, help='result window scale')
    parser.add_argument('--draw_keypoints_numbers', action='store_true', help='draw numbers of keypoints')
    parser.add_argument('--use_image_as_video', action='store_true', help='use input image as video')

    parser.add_argument('-mp', '--model_file_posenet', default='models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite', help='posenet model file')
    parser.add_argument('-mm', '--model_file_movenet', default='models/singlepose-lightning-tflite-int8.tflite', help='movenet model file')
    parser.add_argument('-my', '--model_file_yolov8', default='models/yolov8n-pose_full_integer_quant_320.tflite', help='yolov8 model file')
    parser.add_argument('-mg', '--model_file_yoga', default='models/yoga_classifier.tflite', help='yoga classifier model file')

    parser.add_argument('--thresh_det', type=float, default=0.3, help='threshold of detection')
    parser.add_argument('--thresh_nms', type=float, default=0.4, help='threshold of NMS')
    parser.add_argument('--num_threads', type=int, default=4, help='num of threads in tflite interpreter')

    parser.add_argument('--thresh_angle', type=float, default=np.pi/4, help='threshold of apine angle for fall detection')

    parser.add_argument('--mp_legacy_model_complexity', type=int, default=0, choices=[0,1,2], help='[mediapipe legacy] model_complexity')
    parser.add_argument('--mp_legacy_min_detection_confidence', type=float, default=0.5, help='[mediapipe legacy] min_detection_confidence')
    parser.add_argument('--mp_legacy_min_tracking_confidence', type=float, default=0.5, help='[mediapipe legacy] min_tracking_confidence')

    parser.add_argument('--mp_model_asset_path', type=str, default='models/pose_landmarker_lite.task', help='[mediapipe] model_asset_path')
    parser.add_argument('--mp_delegate', type=int, default=None, choices=[0,1], help='[mediapipe] delegate 0:CPU, 1:GPU')
    parser.add_argument('--mp_num_poses', type=int, default=4, help='[mediapipe] num_poses')
    parser.add_argument('--mp_min_pose_detection_confidence', type=float, default=0.5, help='[mediapipe] min_pose_detection_confidence')
    parser.add_argument('--mp_min_pose_presence_confidence', type=float, default=0.5, help='[mediapipe] min_pose_presence_confidence')
    parser.add_argument('--mp_min_tracking_confidence', type=float, default=0.5, help='[mediapipe] min_tracking_confidence')

    parser.add_argument('--stgcn_type', type=int, default=0, choices=[0,1,2], help='[STGCN] model type')
    parser.add_argument('--stgcn_frames', type=int, default=100, help='[STGCN] num of frames to input the model')
    parser.add_argument('--stgcn_freq', type=int, default=10, help='[STGCN] frames frequency to invoke recognition')

    parser.add_argument('--show_tflite_runtime_version', action='store_true', help='print tflite_runtime version')

    args = parser.parse_args()

    if args.show_tflite_runtime_version:
        import tflite_runtime
        print('tflite_runtime version =', tflite_runtime.__version__)

    # init pose estimator
    param = None
    if args.type == 'mediapipe_legacy':
        param = PoseEstimationMediapipeLegacyParam()
        param.model_complexity = args.mp_legacy_model_complexity
        param.min_detection_confidence = args.mp_legacy_min_detection_confidence
        param.min_tracking_confidence = args.mp_legacy_min_tracking_confidence
    elif args.type == 'mediapipe':
        param = PoseEstimationMediapipeParam()
        param.model_asset_path = args.mp_model_asset_path
        param.delegate = args.mp_delegate
        param.num_poses = args.mp_num_poses
        param.min_pose_detection_confidence = args.mp_min_pose_detection_confidence
        param.min_pose_presence_confidence = args.mp_min_pose_presence_confidence
        param.min_tracking_confidence = args.mp_min_tracking_confidence
    pose = PoseEstimation(args, type=args.type, num_threads=args.num_threads, param=param, thresh_det=args.thresh_det, thread_nms=args.thresh_nms)

    # init Yoga
    yoga = None
    if args.yoga:
        yoga = YogaClassifier(modelfile=args.model_file_yoga, num_threads=args.num_threads)

    # init ST-GCN
    act = None
    pose_stock = None
    act_f_cnt = 0
    if args.action_recognition:
        act = ActionRecognitionSTGCN(type=args.stgcn_type)
        pose_stock = PoseStockSingle(n_frame=args.stgcn_frames, n_joint=act.num_joint)

    # prepare input
    cap = None
    if args.input is not None:
        if args.input.isdigit():
            cap = cv2.VideoCapture(int(args.input))
            if not cap.isOpened():
                print('fail to open camera', args.input)
                sys.exit(-1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            ext = os.path.splitext(args.input)[1][1:]
            if ext == 'mp4' or ext == 'avi' or ext == 'mpg':
                cap = cv2.VideoCapture(args.input)
                if not cap.isOpened():
                    print('fail to open video', args.input)
                    sys.exit(-1)

    # prepare 3D view
    if args.show_3d:
        PoseEstimation.prepareShow3d()

    # main loop
    print('')
    b_once = True
    b_draw = True
    pre_msg_act = []
    waitmsec = 1
    timestamp = 0
    while 1:
        t0 = time.perf_counter()

        if args.input is None:
            # random input data
            shape = pose.getInputShape()
            if shape is None:
                shape = (480, 640, 3)
            img = np.array(np.random.random_sample(shape) * 255, dtype=np.uint8)
            b_draw = False
        elif cap is not None:
            ret, img = cap.read()
            if not ret:
                #print('fail to capture')
                break
            b_once = False
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            img = cv2.imread(args.input)
            if img is None:
                print('fail to open', args.input)
                break
            if args.use_image_as_video:
                b_once = False
                timestamp += 1

        t1 = time.perf_counter()

        norm_scale, norm_offset = pose.setInput(img, timestamp)
        t2 = time.perf_counter()

        time_invoke = []
        for _ in range(3 if b_once and args.type != 'mediapipe' else 1):
            ti0 = time.perf_counter()
            pose.predict()
            ti1 = time.perf_counter()
            time_invoke.append(ti1 - ti0)
        t3 = time.perf_counter()

        # results: 2D, 3D keypoints
        pose_keypoints2d, pose_keypoints3d = pose.getResults(norm_scale, norm_offset)
        
        # message to display results
        msgs = [[] for _ in range(len(pose_keypoints2d))]
        msg_colors = [(255, 100, 100) for _ in range(len(pose_keypoints2d))]

        # calculate spine angle
        angles, spine_points3d = None, None
        if args.detect_spine_angle and pose_keypoints3d is not None:
            angles, spine_points3d = PoseEstimation.calculateSpineAngle(pose_keypoints3d)
            for i, angle in enumerate(angles):
                msgs[i].append('Spine Angle:')
                msgs[i].append(' {:.1f} degree'.format(angle * 180 / np.pi))
                if abs(angle) >= args.thresh_angle:
                    msg_colors[i] = (0, 0, 255)
                    msgs[i].append('  Fall Down !')

        # Yoga
        yoga_results = None
        if yoga is not None:
            yoga_results = []
            for i, keypoints in enumerate(pose_keypoints2d):
                res = yoga.predict(keypoints)
                yoga_results.append(res)
                msgs[i].append('Yoga:')
                if res is not None:
                    for order, (label, score) in enumerate(res):
                        msgs[i].append('<{}> "{}" {:.3f}'.format(order + 1, label, score))

        # action recognition by ST-GCN
        act_topn = None
        if pose_stock is not None:
            if len(pose_keypoints2d) > 0:   # use index 0 person. can not be applied for multi person.
                input_act, b_full = pose_stock.append(np.array(pose_keypoints2d[0]), img.shape)
                if b_full:
                    act_f_cnt = (act_f_cnt + 1) % args.stgcn_freq
                    if act_f_cnt == 0:
                        act_topn = act.predict(input_act)
                        _start_idx = len(msgs[0])
                        msgs[0].append('Action:')
                        for order, (_, label, score) in enumerate(act_topn):
                            msgs[0].append('<{}> "{}" : {}'.format(order + 1, label, score))
                        _end_idx = len(msgs[0])
                        pre_msg_act = msgs[0][_start_idx:_end_idx]
                    else:
                        # show the result on the following frames
                        msgs[0].extend(pre_msg_act)
        
        # print message to stdout
        for i, msg in enumerate(msgs):
            if len(msg) > 0:
                print('Person {}'.format(i))
                for ms in msg:
                    print(ms)
        
        t4 = time.perf_counter()

        # draw results
        if b_draw:
            pose.draw2d(img, pose_keypoints2d, msgs, msg_colors, b_draw_numbers=args.draw_keypoints_numbers)
            if args.show_scale != 1.0:
                img = cv2.resize(img, None, fx=args.show_scale, fy=args.show_scale)
            cv2.imshow('2D Pose', img)
            
            key = 0
            while True:
                if args.show_3d and pose_keypoints3d is not None:
                    PoseEstimation.show3d(pose_keypoints3d)
                
                key = cv2.waitKey(waitmsec)
                if key == 27:   # ESC
                    break
                if key == ord('s'): # save screenshot images
                    cv2.imwrite('screenshot.png', img)
                    if args.show_3d:
                        PoseEstimation.save3d('screenshot3d.png')
                if (not b_once) or (key != -1):
                    break
            if key == 27:
                break

        t5 = time.perf_counter()

        print('total {:.4f}, input {:.4f}, norm {:.4f}, invoke {:4f}, post {:.4f}, show {:.4f}'
              .format(t5-t0, t1-t0, t2-t1, time_invoke[-1], t4-t3, t5-t4))

        if len(time_invoke) > 1:
            print('invoke ', end='')
            for t in time_invoke:
                print('{:.4f},'.format(t))
            print('')

        if b_once:
            break
