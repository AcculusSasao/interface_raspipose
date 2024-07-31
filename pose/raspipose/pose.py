# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import List, Tuple, Any
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite.Interpreter

from .pose_utils import POSE17_JOINTS, POSE33_JOINTS, dequantize, YoloV8PosePost
from .posenet.posenet import Posenet

def normalize_image(bgr_img: np.array, target_width: int, target_height: int, dtype: int,
                    mean: float = 0, std : float = 1, quantization = None) -> np.array:
    if target_width is None:
        img = bgr_img
    else:
        if target_width != target_height:
            raise ValueError('target_width must be same with target_height.')
        h, w = bgr_img.shape[:2]
        if h <= w:
            srcimg = np.zeros((w, w, 3), dtype=np.uint8)
            e = (w - h) // 2
            srcimg[e:e+h] = bgr_img
            offset = [e, 0]
        else:
            srcimg = np.zeros((h, h, 3), dtype=np.uint8)
            e = (h - w) // 2
            srcimg[:, e:e+w] = bgr_img
            offset = [0, e]
        img = cv2.resize(srcimg, dsize=(target_width, target_height))
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = ((data - mean) / std)
    if quantization is not None:
        sc, zp = quantization
        if sc != 0:
            data = data / sc + zp
    data = data.astype(dtype)
    if target_width is None:
        scale = 1
        offset = np.zeros((2))
    else:
        scale = img.shape[0] / srcimg.shape[0]
        offset = np.array(offset)
    return data[np.newaxis, :], scale, offset

# https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
@dataclass
class PoseEstimationMediapipeLegacyParam():
    static_image_mode: bool = False
    model_complexity: int = 0   # [0,1,2]
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

# https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/PoseLandmarkerOptions
@dataclass
class PoseEstimationMediapipeParam():
    model_asset_path: str = 'pose_landmarker_lite.task'   # ['pose_landmarker_lite.task', 'pose_landmarker_full.task', 'pose_landmarker_heavy.task']
    model_asset_buffer: bytes = None
    delegate = None         # None or 0(CPU), 1(GPU)
    running_mode = 'VIDEO'  # 'IMAGE', 'VIDEO', 'LIVE_STREAM'
    num_poses: int = 4
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_segmentation_masks: bool = False

class PoseEstimation:
    _b_first_show3d = True
    _rotate3d_diff = 0
    _rotate3d_cur = 0
    
    def __init__(self, args, type: str, num_threads: int = 4, param: Any = None, thresh_det: float = 0.3, thread_nms: float = 0.4) -> None:
        self.mean, self.std = 0, 1
        self.type = type
        if 'mediapipe' in self.type:
            import mediapipe as mp
            self._mp = mp
            if self.type == 'mediapipe_legacy':
                p: PoseEstimationMediapipeLegacyParam = param
                if p is None:
                    p = PoseEstimationMediapipeLegacyParam()
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=p.static_image_mode,
                    model_complexity=p.model_complexity,
                    smooth_landmarks=p.smooth_landmarks,
                    enable_segmentation=p.enable_segmentation,
                    smooth_segmentation=p.smooth_segmentation,
                    min_detection_confidence=p.min_detection_confidence,
                    min_tracking_confidence=p.min_tracking_confidence,
                )
            elif self.type == 'mediapipe':
                p: PoseEstimationMediapipeParam = param
                if p is None:
                    p = PoseEstimationMediapipeParam()
                running_mode = p.running_mode
                if running_mode == 'IMAGE':
                    running_mode = mp.tasks.vision.RunningMode.IMAGE
                elif running_mode == 'VIDEO':
                    running_mode = mp.tasks.vision.RunningMode.VIDEO
                elif running_mode == 'LIVE_STREAM':
                    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM
                delegate = p.delegate
                if delegate == 0:
                    delegate = mp.tasks.BaseOptions.Delegate.CPU
                elif delegate == 1:
                    delegate = mp.tasks.BaseOptions.Delegate.GPU
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path = p.model_asset_path,
                        model_asset_buffer = p.model_asset_buffer,
                        delegate = delegate
                        ),
                    running_mode = running_mode,
                    num_poses = p.num_poses,
                    min_pose_detection_confidence = p.min_pose_detection_confidence,
                    min_pose_presence_confidence = p.min_pose_presence_confidence,
                    min_tracking_confidence = p.min_tracking_confidence,
                    output_segmentation_masks = p.output_segmentation_masks,
                    )
                self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                pass
        else:
            if self.type == 'posenet':
                modelfile = args.model_file_posenet
                self.mean = self.std = 127.5
            elif self.type == 'movenet':
                modelfile = args.model_file_movenet
            elif self.type == 'yolov8':
                modelfile = args.model_file_yolov8
                self.std = 255
                self.yolo = YoloV8PosePost(input_size=320)
            self.thresh_det = thresh_det
            self.thresh_nms = thread_nms
            self._interpreter = tflite.Interpreter(model_path=modelfile, num_threads=num_threads)
            self._interpreter.allocate_tensors()

            print('\nmodel input:')
            self.input_details = self._interpreter.get_input_details()
            for detail in self.input_details:
                print(detail)
            print('\nmodel output:')
            self.output_details = self._interpreter.get_output_details()
            for detail in self.output_details:
                print(detail)
            self.input_shape = self.input_details[0]['shape']
            self.input_index = self.input_details[0]['index']
            self.input_type = self.input_details[0]['dtype']
            self.input_quantization = self.input_details[0]['quantization']
        
    def getInputShape(self) -> Tuple[int]:
        if 'mediapipe' in self.type:
            return None
        else:
            return self.input_shape[1:4]
    
    def setInput(self, bgrimg: np.array, timestamp_ms: int = None) -> Tuple[float, np.array]:
        if 'mediapipe' in self.type:
            self._input_bgrimg = bgrimg
            self._timestamp_ms = timestamp_ms
            return 1, np.zeros((2))
        else:
            input_data, norm_scale, norm_offset = normalize_image(bgrimg, self.input_shape[2], self.input_shape[1], self.input_type,
                mean=self.mean, std=self.std, quantization=self.input_quantization)
            self._interpreter.set_tensor(self.input_index, input_data)
            return norm_scale, norm_offset
    
    def predict(self) -> None:
        if self.type == 'mediapipe_legacy':
            self._results = self._pose.process(cv2.cvtColor(self._input_bgrimg, cv2.COLOR_BGR2RGB))
        elif self.type == 'mediapipe':
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=self._input_bgrimg)
            self._results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            self._interpreter.invoke()
    
    def getResults(self, norm_scale: float, norm_offset: np.array) -> List[List[List[float]]]:
        pose_keypoints2d = []
        pose_keypoints3d = None
        if self.type == 'posenet':
            outs = []
            for output in self.output_details:
                out = self._interpreter.get_tensor(output['index'])
                outs.append(out)
            keypoints_with_scores = Posenet.process_output(
                outs[0].squeeze(axis=0),
                outs[1].squeeze(axis=0)
            )
            keypoints = []
            for kps in keypoints_with_scores:
                y, x, s = kps
                x = x * self.input_shape[2] / norm_scale - norm_offset[1]
                y = y * self.input_shape[1] / norm_scale - norm_offset[0]
                keypoints.append([x, y, s])
            pose_keypoints2d.append(keypoints)

        elif self.type == 'movenet':
            out = self._interpreter.get_tensor(self.output_details[0]['index'])[0][0]
            keypoints = []
            for kps in out:
                y, x, s = kps
                x = x * self.input_shape[2] / norm_scale - norm_offset[1]
                y = y * self.input_shape[1] / norm_scale - norm_offset[0]
                keypoints.append([x, y, s])
            pose_keypoints2d.append(keypoints)

        elif self.type == 'yolov8':
            out = self._interpreter.get_tensor(self.output_details[0]['index'])
            if len(self.output_details) == 1:
                out = out[0]
                out = dequantize(out, self.output_details[0]['quantization'])
                out = out.transpose((1, 0))
            else:
                out = self.yolo.postprocess_for_separated_outputs(self._interpreter, self.output_details,
                                                                  thresh_det=self.thresh_det, thresh_nms=self.thresh_nms)
            for d in out:
                if d[4] >= self.thresh_det:
                    keypoints = []
                    for i in range(17):
                        x = d[5 + i*3 + 0]
                        y = d[5 + i*3 + 1]
                        s = d[5 + i*3 + 2]
                        x = x / norm_scale - norm_offset[1]
                        y = y / norm_scale - norm_offset[0]
                        keypoints.append([x, y, s])
                    pose_keypoints2d.append(keypoints)
        
        elif self.type == 'mediapipe_legacy':
            pose_keypoints3d = []
            results = self._results
            bgr = self._input_bgrimg
            if results.pose_landmarks is not None:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    x = landmark.x * bgr.shape[1]
                    y = landmark.y * bgr.shape[0]
                    s = landmark.visibility
                    keypoints.append([x, y, s])
                pose_keypoints2d.append(keypoints)
            if results.pose_world_landmarks is not None:
                keypoints = []
                for landmark in results.pose_world_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    s = landmark.visibility
                    keypoints.append([x, y, z, s])
                pose_keypoints3d.append(keypoints)

        elif self.type == 'mediapipe':
            pose_keypoints3d = []
            results = self._results
            bgr = self._input_bgrimg
            if results.pose_landmarks is not None:
                for pose_landmark in results.pose_landmarks:
                    keypoints = []
                    for landmark in pose_landmark:
                        x = landmark.x * bgr.shape[1]
                        y = landmark.y * bgr.shape[0]
                        s = landmark.visibility     # or landmark.presence
                        keypoints.append([x, y, s])
                    pose_keypoints2d.append(keypoints)
            if results.pose_world_landmarks is not None:
                for pose_landmark in results.pose_world_landmarks:
                    keypoints = []
                    for landmark in pose_landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        s = landmark.visibility
                        keypoints.append([x, y, z, s])
                    pose_keypoints3d.append(keypoints)

        return pose_keypoints2d, pose_keypoints3d
    
    @staticmethod
    def draw2d(img: np.array, pose_keypoints2d: List[List[List[float]]],
               msgs: List[List[str]] = None, msg_colors: List[Tuple[int,int,int]] = None,
               b_draw_numbers: bool = False
               ) -> None:
        d = 2
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        dify = 24
        for kidx, keypoints in enumerate(pose_keypoints2d):
            th = 1e-5

            joints = POSE17_JOINTS if len(keypoints) == 17 else POSE33_JOINTS
            for joint in joints:
                if keypoints[joint[0]][2] < th or keypoints[joint[1]][2] < th:
                    continue
                x0, y0 = keypoints[joint[0]][:2]
                x1, y1 = keypoints[joint[1]][:2]
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), thickness=6)

            for kps in keypoints:
                x, y, s = kps
                if s < th:
                    continue
                cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), thickness=4)
            if b_draw_numbers:
                for kidx, kps in enumerate(keypoints):
                    x, y, s = kps
                    if s < th:
                        continue
                    cv2.putText(img, str(kidx), (int(x+d), int(y+d)), fontFace, fontScale, (200,200,200), thickness)
                    cv2.putText(img, str(kidx), (int(x), int(y)), fontFace, fontScale, (20,20,20), thickness)

            if msgs is None or msg_colors is None or kidx >= len(msgs):
                continue
            msg = msgs[kidx]
            msg_color = msg_colors[kidx]
            x = int(np.mean([kp[0] for kp in keypoints]))
            y = int(np.mean([kp[1] for kp in keypoints]))
            for ms in msg:
                cv2.putText(img, ms, (x+d, y+d), fontFace, fontScale, (0, 0, 0), thickness)
                cv2.putText(img, ms, (x, y), fontFace, fontScale, msg_color, thickness)
                y += dify

    @classmethod
    def prepareShow3d(cls, title: str = '3D Pose', axis_labels = ['X', 'Y', 'Z'],
                      rotate3d_diff: float = np.pi / 16) -> None:
        import matplotlib
        matplotlib.use('TkAgg')	# 'tkinter' is needed. Do 'apt install python3-tk tk-dev' before installing pyenv.
        import matplotlib.pyplot as plt
        cls._plt = plt
        fig = plt.figure(title)
        cls._ax = fig.add_subplot(projection='3d')
        cls._ax.view_init(elev=-90, azim=-90, vertical_axis='z')
        cls._ax_labels = axis_labels
        plt.ion()
        cls._plt_markers = ('o',',','*','v','^','<','>','8','p','+','x','D')
        cls._plt_colors = ("g", "r", "b", "c", "m", "y", "k")
        cls._rotate3d_diff = rotate3d_diff
        cls._rotate3d_cur = 0

    @classmethod
    def show3d(cls, pose_keypoints3d: List[List[List[float]]], interval_sec: float = 0.0001) -> None:
        if pose_keypoints3d is None:
            return
        cls._ax.cla()
        cls._ax.set_xlabel(cls._ax_labels[0])
        cls._ax.set_ylabel(cls._ax_labels[1])
        cls._ax.set_zlabel(cls._ax_labels[2])
        cls._ax.set_xlim(-1, 1)
        cls._ax.set_ylim(-1, 1)
        cls._ax.set_zlim(-1, 1)
        # rotation matrix to rotate the display
        rot = Rotation.from_rotvec(np.array([0, cls._rotate3d_cur, 0])).as_matrix().T
        
        for kidx, keypoints in enumerate(pose_keypoints3d):
            _keypoints = np.dot(np.array(keypoints)[:,:3], rot)
            
            x = [kp[0] for kp in _keypoints]
            y = [kp[1] for kp in _keypoints]
            z = [kp[2] for kp in _keypoints]
            
            color = cls._plt_colors[kidx % len(cls._plt_colors)]
            cls._ax.scatter(x, y, z, marker=cls._plt_markers[kidx % len(cls._plt_markers)], color=color)
            joints = POSE17_JOINTS if len(keypoints) == 17 else POSE33_JOINTS
            for joint in joints:
                cls._ax.plot(
                    [_keypoints[joint[0]][0], _keypoints[joint[1]][0]],
                    [_keypoints[joint[0]][1], _keypoints[joint[1]][1]],
                    [_keypoints[joint[0]][2], _keypoints[joint[1]][2]],
                    color=color
                    )
        cls._plt.draw()
        cls._plt.pause(interval_sec + (1 if cls._b_first_show3d else 0))
        cls._b_first_show3d = False
        cls._rotate3d_cur += cls._rotate3d_diff

    @classmethod
    def save3d(cls, filename: str) -> None:
        cls._plt.savefig(filename)
    
    @staticmethod
    def calculateSpineAngle(pose_keypoints3d: List[List[List[float]]]) -> Tuple[List[float],List[List[np.array]]]:
        VEC_STAND_UP_STRAIGHT = np.array([0, -1, 0])
        angles = []
        spine_points3d = []
        if pose_keypoints3d is None:
            return angles, spine_points3d
        for keypoints in pose_keypoints3d:
            if len(keypoints) != 33:
                print('calculateSpineAngle supports mediapipe only.')
                return angles, spine_points3d
            top = (np.array(keypoints[11][:3]) + np.array(keypoints[12][:3])) / 2
            bot = (np.array(keypoints[23][:3]) + np.array(keypoints[24][:3])) / 2
            spine_points3d.append([top, bot])
            vec = top - bot
            vec = vec / np.linalg.norm(vec)
            angle = np.arccos(np.dot(vec, VEC_STAND_UP_STRAIGHT))
            angles.append(angle)
        return angles, spine_points3d
