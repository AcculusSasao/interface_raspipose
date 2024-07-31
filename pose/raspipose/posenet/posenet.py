# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code to run a pose estimation with a TFLite PoseNet model."""
import numpy as np

class Posenet(object):
  """A wrapper class for a Posenet TFLite pose estimation model."""

  @staticmethod
  def _sigmoid(x: np.ndarray) -> float:
    return 1 / (1 + np.exp(-x))

  @staticmethod
  def process_output(heatmap_data: np.ndarray,
                      offset_data: np.ndarray) -> np.ndarray:
    """Post-process the output of Posenet TFLite model.

    Args:
      heatmap_data: heatmaps output from Posenet. [height_resolution,
        width_resolution, 17]
      offset_data: offset vectors (XY) output from Posenet. [height_resolution,
        width_resolution, 34]

    Returns:
      An array of shape [17, 3] representing the keypoint absolute coordinates
      and scores.
    """
    joint_num = heatmap_data.shape[-1]
    keypoints_with_scores = np.zeros((joint_num, 3), np.float32)
    scores = Posenet._sigmoid(heatmap_data)

    for idx in range(joint_num):
      joint_heatmap = heatmap_data[..., idx]
      x, y = np.unravel_index(
          np.argmax(scores[:, :, idx]), scores[:, :, idx].shape)
      max_val_pos = np.squeeze(
          np.argwhere(joint_heatmap == np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)

      keypoints_with_scores[idx, 0] = (
          remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], idx]) / 257
      keypoints_with_scores[idx, 1] = (
          remap_pos[1] +
          offset_data[max_val_pos[0], max_val_pos[1], idx + joint_num]) / 257
      keypoints_with_scores[idx, 2] = scores[x, y, idx]

    return keypoints_with_scores
