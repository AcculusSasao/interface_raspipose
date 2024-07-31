# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import numpy as np

POSE17_KEYPOINTS_STR = (
    "nose",         # 0
    "leftEye",      # 1
    "rightEye",     # 2
    "leftEar",      # 3
    "rightEar",     # 4
    "leftShoulder", # 5
    "rightShoulder",# 6
    "leftElbow",    # 7
    "rightElbow",   # 8
    "leftWrist",    # 9
    "rightWrist",   # 10
    "leftHip",      # 11
    "rightHip",     # 12
    "leftKnee",     # 13
    "rightKnee",    # 14
    "leftAnkle",    # 15
    "rightAnkle"    # 16
)
POSE17_JOINTS = (
    (0, 5),
    (0, 6),
    (1, 2),
    (3, 1),
    (4, 2),
    (11, 5),
    (7, 5),
    (7, 9),
    (11, 13),
    (13, 15),
    (12, 6),
    (8, 6),
    (8, 10),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
)
POSE33_JOINTS = (
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (10, 9),
    (12, 11),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (16, 20),
    (16, 22),
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
    (15, 19),
    (15, 21),
    (12, 24),
    (11, 23),
    (24, 23),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
)

def dequantize(data: np.array, quantization) -> np.array:
    if quantization is None or quantization[0] == 0:
        return data
    sc, zp = quantization
    return (data.astype(np.float32) - zp) * sc

def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))

class YoloV8PosePost():
    def __init__(self, input_size: int = 320) -> None:
        if input_size == 320:
            self._strides = [8, 16, 32]
            self._dims = [(40, 40), (20, 20), (10, 10)]
        else:
            raise ValueError('Input size supports 320 only.')
        self._input_size = input_size
        self._grid_cell_offset = 0.5
        self.anchors, self.strides = self.make_anchors()

    def make_anchors(self):
        anchor_points, stride_tensor = [], []
        for dim, stride in zip(self._dims, self._strides):
            h, w = dim
            sx = np.arange(w) + self._grid_cell_offset
            sy = np.arange(h) + self._grid_cell_offset
            sx, sy = np.meshgrid(sy, sx)
            a = np.stack((sx, sy), axis=-1).reshape(-1, 2)
            anchor_points.append(a)
            stride_tensor.append(np.full((h * w, 1), stride))
        return np.concatenate(anchor_points).T, np.concatenate(stride_tensor).T

    def postprocess_for_separated_outputs(self, interpreter, output_details, 
                                          thresh_det=0.25, thresh_nms=0.3, b_out_flatten=True):
        candidates = self.decode_for_separated_outputs(interpreter, output_details, thresh_det)
        humans = self.nms(candidates, thresh_nms)
        if b_out_flatten:
            results = []
            for (bb_conf, xyxy, kps_y, kps_x, kps_conf) in humans:
                res = xyxy
                res.append(bb_conf)
                for y, x, s in zip(kps_y, kps_x, kps_conf):
                    res.extend([y, x, s])
                results.append(res)
            return results
        return humans

    def decode_for_separated_outputs(self, interpreter, output_details, thresh_det=0.25):
        # seek for index numbers of keypoints and bbox scores
        out_idx_keypoints = None
        list_shape1 = []
        for idx, output_detail in enumerate(output_details):
            shape = output_detail['shape']
            if shape[-1] == 51:
                out_idx_keypoints = idx
            elif shape[-1] == 1:
                list_shape1.append([shape[-2], idx])
        list_shape1 = sorted(list_shape1, key=lambda x: -x[0])
        out_idxes_bbox_conf = [x[1] for x in list_shape1]
        if out_idx_keypoints is None or len(out_idxes_bbox_conf) < 1:
            raise ValueError('output format is unknown')

        # get confident bboxes
        outs_conf = []
        for idx in out_idxes_bbox_conf:
            out = interpreter.get_tensor(output_details[idx]['index'])[0]
            out = dequantize(out, output_details[idx]['quantization'])
            outs_conf.append(out)
        bbox_conf = np.concatenate(outs_conf, axis=0)
        bbox_conf = sigmoid_numpy(bbox_conf).flatten()
        bbox_idxes = np.where(bbox_conf >= thresh_det)[0]

        out_kps = interpreter.get_tensor(output_details[out_idx_keypoints]['index'])[0]
        candidates = []
        for bbidx in bbox_idxes:
            bb_conf = bbox_conf[bbidx]

            kps = out_kps[bbidx]
            kps = dequantize(kps, output_details[out_idx_keypoints]['quantization'])
            kps_conf = sigmoid_numpy(kps[2::3])
            kps_y = kps[0::3]
            kps_y = (kps_y * 2.0 + (self.anchors[0, bbidx] - 0.5)) * self.strides[0, bbidx]
            kps_x = kps[1::3]
            kps_x = (kps_x * 2.0 + (self.anchors[1, bbidx] - 0.5)) * self.strides[0, bbidx]

            valid_kp_idxes = kps_conf > 0
            xyxy = [
                np.min(kps_x[valid_kp_idxes]),
                np.min(kps_y[valid_kp_idxes]),
                np.max(kps_x[valid_kp_idxes]),
                np.max(kps_y[valid_kp_idxes]),
            ]

            cand = [bb_conf, xyxy, kps_y, kps_x, kps_conf]
            candidates.append(cand)
        candidates = sorted(candidates, key=lambda x: -x[0])
        return candidates
    
    def nms(self, candidates, thresh_nms=0.3):
        num = len(candidates)
        valid = np.ones((num), dtype=np.int8)
        for i in range(num):
            if not valid[i]:
                continue
            a = candidates[i][1]
            ma = abs((a[2] - a[0]) * (a[3] - a[1]))
            for j in range(i+1, num):
                if not valid[j]:
                    continue
                b = candidates[j][1]
                # IoU
                x0 = min(a[0], b[0])
                y0 = min(a[1], b[1])
                x1 = max(a[2], b[2])
                y1 = max(a[3], b[3])
                inter = abs(max((x1 - x0, 0)) * max((y1 - y0), 0))
                if inter == 0:
                    continue
                mb = abs((b[2] - b[0]) * (b[3] - b[1]))
                iou = inter / (ma + mb - inter)
                if iou < thresh_nms:
                    continue
                valid[j] = 0
        return [c for v, c in zip(valid, candidates) if v]
