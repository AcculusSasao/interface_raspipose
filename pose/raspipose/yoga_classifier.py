# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import numpy as np
from typing import List, Tuple
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite.Interpreter

YOGA_LABELS=(
    'chair',
    'cobra',
    'dog',
    'tree',
    'warrior',
)

class YogaClassifier:
    def __init__(self, modelfile: str, num_threads: int = 4, min_kp_score: float = 0.1) -> None:
        self._min_kp_score = min_kp_score
        self._interpreter = tflite.Interpreter(model_path=modelfile, num_threads=num_threads)
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()
        self._input_index = input_details[0]['index']
        output_details = self._interpreter.get_output_details()
        self._output_index = output_details[0]['index']
    
    def predict(self, keypoints: List[Tuple[float,float,float]], topn: int = 3
                ) -> List[Tuple[str, float]]:
        if keypoints is None or len(keypoints) != 17:
            print('Yoga classifier does not support {} keypoints'.format(len(keypoints)))
            return None
        score = min([k[2] for k in keypoints])
        if score < self._min_kp_score:
            print('Some keypoints are not detected.')
            return None
        
        input = np.array([[k[1], k[0], k[2]] for k in keypoints]).flatten().astype(np.float32)
        input = input[np.newaxis, :]
        
        self._interpreter.set_tensor(self._input_index, input)
        self._interpreter.invoke()
        
        output = self._interpreter.get_tensor(self._output_index).flatten()

        order = sorted(range(len(output)), key=lambda i: output[i], reverse=True)

        # return [ ( label, score ) ]
        result = []
        for i in range(topn):
            j = order[i]
            result.append((YOGA_LABELS[j], output[j]))
        return result
