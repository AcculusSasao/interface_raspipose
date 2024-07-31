# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
# using https://github.com/open-mmlab/mmskeleton
import torch
import numpy as np
from typing import Tuple, List
import sys, os

_this_dir = os.path.dirname(__file__)
sys.path.append(_this_dir + '/mmskeleton')
from .mmskeleton.mmskeleton.models.backbones.st_gcn_aaai18 import ST_GCN_18

class PoseStockSingle:
    def __init__(self, n_frame: int, n_joint: int = 18) -> None:
        if n_joint != 18:
            raise ValueError('PoseStockSingle supports 18 joints only.')
        self.n_frame = n_frame
        self.n_joint = n_joint
        # (batch, channel, frame, joint, person)
        self.ary = np.zeros((1, 3, self.n_frame, self.n_joint, 1))
        self.clear()
    
    def clear(self) -> None:
        self.ary[:] = 0
        self.n_append = 0
    
    def get(self) -> Tuple[np.array, bool]:
        b_full = self.n_append >= self.n_frame
        return self.ary, b_full
    
    def append(self, keypoints: np.array, img_shape: Tuple[int, int]) -> Tuple[np.array, bool]:
        kps = self.convert_kps(keypoints)
        kps[:, 0] = kps[:, 0] / img_shape[1] - 0.5
        kps[:, 1] = kps[:, 1] / img_shape[0] - 0.5
        elem = np.reshape(kps.T, (1, 3, 1, self.n_joint, 1))
        newary = self.ary[:, :, 1:, :, :]
        newary = np.append(newary, elem, axis=2)
        self.ary = newary
        self.n_append += 1
        return self.get()
    
    def convert_kps(self, kps: np.array) -> np.array:
        if len(kps) == self.n_joint:
            return kps
        elif len(kps) == 17:    # coco format
            r = np.zeros((self.n_joint, 3))
            r[0] = kps[0]
            r[1] = (kps[5] + kps[6]) / 2
            r[2] = kps[6]
            r[3] = kps[8]
            r[4] = kps[10]
            r[5] = kps[5]
            r[6] = kps[7]
            r[7] = kps[9]
            r[8] = kps[12]
            r[9] = kps[14]
            r[10] = kps[16]
            r[11] = kps[11]
            r[12] = kps[13]
            r[13] = kps[15]
            r[14] = kps[2]
            r[15] = kps[1]
            r[16] = kps[4]
            r[17] = kps[3]
            return r
        raise(ValueError("The number of keypoints is not supported."))

class ActionRecognitionSTGCN:
    _checkpoints = (
        'st_gcn.kinetics-6fa43f73.pth',
        'st_gcn.ntu-xsub-300b57d4.pth',
        'st_gcn.ntu-xview-9ba67746.pth',
    )

    def __init__(self, type: int = 0) -> None:
        if type is not None:
            self.load_model(type)

    def load_model(self, type: int = 0) -> bool:
        self.labels = []
        if type == 0:
            self.in_channels = 3
            self.num_class = 400
            self.num_joint = 18
            self._model = ST_GCN_18(
                in_channels = self.in_channels,
                num_class = self.num_class,
                graph_cfg = {'layout': 'openpose', 'strategy': 'spatial'},
                edge_importance_weighting = True
            )
            self.labels = ActionRecognitionSTGCN.load_models_txt()
            if len(self.labels) != self.num_class:
                print('Num of labels {} != num of class {}'.format(len(self.labels), self.num_class))
        elif type == 1:
            self.in_channels = 3
            self.num_class = 60
            self.num_joint = 25
            self._model = ST_GCN_18(
                in_channels = self.in_channels,
                num_class = self.num_class,
                graph_cfg = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                edge_importance_weighting = True,
                dropout = 0.5,
            )
        elif type == 2:
            self.in_channels = 3
            self.num_class = 60
            self.num_joint = 25
            self._model = ST_GCN_18(
                in_channels = self.in_channels,
                num_class = self.num_class,
                graph_cfg = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                edge_importance_weighting = True,
                dropout = 0.5,
            )
        else:
            return False
        self._model.load_state_dict(torch.load(_this_dir + "/mmskeleton/checkpoints/" + self._checkpoints[type]))
        self._model.eval()
        return True

    '''
    NN input: (batch, channel, frame, joint, person)
        channel = (x, y, score)
    NN output:(batch, class)
    '''
    def predict(self, input: np.array, topn: int = 3) -> List[Tuple[int, str, float]]:
        input_tensor = torch.from_numpy(input).float()
        output_tensor = self._model(input_tensor)
        output = output_tensor.to('cpu').detach().numpy().copy().flatten()
        maxidxes = np.argsort(output)[::-1]
        # Tuple[index, label, score]
        return [(idx, self.labels[idx] if idx < len(self.labels) else None, output[idx])
                for idx in maxidxes[:topn]]

    @staticmethod
    def load_models_txt(txtfile: str = _this_dir + '/mmskeleton/deprecated/origin_stgcn_repo/resource/kinetics_skeleton/label_name.txt'
                        ) -> List[str]:
        labels = []
        with open(txtfile, 'r') as f:
            labels = f.read().splitlines()
        return labels

    def export_onnx(self, onnx_file: str, input_shape: Tuple[int, ...]) -> None:
        '''  To avoid 'Gather' error, modify mmskeleton/models/backbones/st_gcn_aaai18.py around line 104 to be:
        before:     x = F.avg_pool2d(x, x.size()[2:])
        after:      x = F.avg_pool2d(x, (75,18))    # in case 300
            ref: https://medium.com/axinc/st-gcn-%E9%AA%A8%E6%A0%BC%E3%81%8B%E3%82%89%E4%BA%BA%E7%89%A9%E3%81%AE%E3%82%A2%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-af3196e38d1f
                '''
        dummy_input = torch.randn(*input_shape, requires_grad=True)
        torch.onnx.export(self._model, dummy_input, onnx_file, verbose=True, opset_version=12)

    def print_model(self) -> None:
        print(self._model)

if __name__ == "__main__":
    type = 0
    nframes = 300
    b_export = 0
    print('usage: {} [type (default:{})] [nframes (default:{})] [b_export]'.format(sys.argv[0], type, nframes))
    if len(sys.argv) > 1:
        type = int(sys.argv[1])
    if len(sys.argv) > 2:
        nframes = int(sys.argv[2])
    if len(sys.argv) > 3:
        b_export = int(sys.argv[3])
    act = ActionRecognitionSTGCN(type=type)
    input_shape = (1, 3, nframes, act.num_joint, 1)
    input = np.random.random(input_shape)
    output = act.predict(input)
    if b_export:
        onnxfile = 'stgcn_type' + str(type) + '.onnx'
        act.export_onnx(onnxfile, input_shape)
    print()
    print('input shape =', input_shape)
    print('output shape =', output.shape)
    if b_export:
        print('onnx output to', onnxfile)
