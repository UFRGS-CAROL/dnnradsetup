# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to work with a classification model."""

import collections
import operator
import os
import re

import numpy as np
import tensorflow
from PIL import Image


class Class(collections.namedtuple('Class', ['id', 'score'])):
    """Represents a single classification, with the following fields:
    .. py:attribute:: id
        The class id.
    .. py:attribute:: score
        The prediction score.
    """


def num_classes(interpreter):
    """Gets the number of classes output by a classification model.
    Args:
        interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
        The total number of classes output by the model.
    """
    return np.prod(interpreter.get_output_details()[0]['shape'])


def get_classification_scores(interpreter):
    """Gets the output (all scores) from a classification model, dequantizing it if necessary.
    Args:
        interpreter: The ``tf.lite.Interpreter`` to query for output.
    Returns:
        The output tensor (flattened and dequantized) as :obj:`numpy.array`.
    """
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.tensor(output_details['index'])().flatten()

    if np.issubdtype(output_details['dtype'], np.integer):
        scale, zero_point = output_details['quantization']
        # Always convert to np.int64 to avoid overflow on subtraction.
        return scale * (output_data.astype(np.int64) - zero_point)

    return output_data


def get_classes_from_scores(scores,
                            top_k=float('inf'),
                            score_threshold=-float('inf')):
    """Gets results from a classification model as a list of ordered classes, based on given scores.
    Args:
        scores: The output from a classification model. Must be flattened and
        dequantized.
        top_k (int): The number of top results to return.
        score_threshold (float): The score threshold for results. All returned
        results have a score greater-than-or-equal-to this value.
    Returns:
        A list of :obj:`Class` objects representing the classification results,
        ordered by scores.
    """
    top_k = min(top_k, len(scores))
    classes = [
        Class(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(classes, key=operator.itemgetter(1), reverse=True)


def get_classes(interpreter, top_k=float('inf'), score_threshold=-float('inf')):
    """Gets results from a classification model as a list of ordered classes.
    Args:
        interpreter: The ``tf.lite.Interpreter`` to query for results.
        top_k (int): The number of top results to return.
        score_threshold (float): The score threshold for results. All returned
        results have a score greater-than-or-equal-to this value.
    Returns:
        A list of :obj:`Class` objects representing the classification results,
        ordered by scores.
    """
    return get_classes_from_scores(get_classification_scores(interpreter), top_k, score_threshold)


def output_tensor(interpreter, i):
    """Gets a model's ith output tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      i (int): The index position of an output tensor.
    Returns:
      The output tensor at the specified position.
    """
    return interpreter.tensor(interpreter.get_output_details()[i]['index'])()


def input_details(interpreter, key):
    """Gets a model's input details by specified key.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      key (int): The index position of an input tensor.
    Returns:
      The input details.
    """
    return interpreter.get_input_details()[0][key]


def get_input_size(interpreter):
    """Gets a model's input size as (width, height) tuple.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor size as (width, height) tuple.
    """
    _, height, width, _ = input_details(interpreter, 'shape')
    return width, height


def input_tensor(interpreter):
    """Gets a model's input tensor view as numpy array of shape (height, width, 3).
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor view as :obj:`numpy.array` (height, width, 3).
    """
    tensor_index = input_details(interpreter, 'index')
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, data):
    """Copies data to a model's input tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` to update.
      data: The input tensor.
    """
    input_tensor(interpreter)[:, :] = data


def set_resized_input(interpreter, resized_image):
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    w, h = resized_image.size
    tensor[:h, :w] = np.reshape(resized_image, (h, w, channel))


def resize_input(image, interpreter):
    width, height = get_input_size(interpreter)
    w, h = image.size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    resized = image.resize((w, h), Image.ANTIALIAS)
    return resized, (scale, scale)


def read_label_file(file_path):
    """Reads labels from a text file and returns it as a dictionary.
    This function supports label files with the following formats:
    + Each line contains id and description separated by colon or space.
        Example: ``0:cat`` or ``0 cat``.
    + Each line contains a description only. The returned label id's are based on
        the row number.
    Args:
        file_path (str): path to the label file.
    Returns:
        Dict of (int, string) which maps label id to description.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
            ret[int(pair[0])] = pair[1].strip()
        else:
            ret[row_number] = pair[0].strip()
    return ret


def create_interpreter(model_file):
    interpreter = tensorflow.lite.Interpreter(model_path=model_file, num_threads=os.cpu_count())
    return interpreter


def get_raw_output(interpreter) -> dict:
    outs_tensors_idxs = list(map(lambda d: d['index'], interpreter.get_output_details()))
    raw_out_dict = {tensor_idx: interpreter.get_tensor(tensor_idx) for tensor_idx in outs_tensors_idxs}
    return raw_out_dict


def save_tensors_to_file(tensors_dict: dict, filename: str):
    np.save(filename, tensors_dict)


def load_tensors_from_file(filename: str):
    return np.load(filename, allow_pickle=True).item()


########################################################################################################################
# For detection

class Object(collections.namedtuple('Object', ['id', 'score', 'bbox'])):
    """Represents a detected object.
    .. py:attribute:: id
        The object's class id.
    .. py:attribute:: score
        The object's prediction score.
    .. py:attribute:: bbox
        A :obj:`BBox` object defining the object's location.
    """

    @staticmethod
    def from_nparray(nparray, input_size=None, img_scale=(1., 1.)):
        return Object(
            id=int(nparray[0]),
            score=nparray[1],
            bbox=BBox.from_nparray(nparray[2:6], input_size, img_scale))

    @property
    def nparray(self):
        return np.concatenate(([self.id, self.score], self.bbox))

    def print(self, labels=None):
        if labels is None:
            labels = {}
        print(labels.get(self.id, self.id))
        print('  id:    ', self.id)
        print('  score: ', self.score)
        print('  bbox:  ', self.bbox)


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """The bounding box for a detected object.
    .. py:attribute:: xmin
        X-axis start point
    .. py:attribute:: ymin
        Y-axis start point
    .. py:attribute:: xmax
        X-axis end point
    .. py:attribute:: ymax
        Y-axis end point
    """
    __slots__ = ()

    @staticmethod
    def from_nparray(nparray, input_size=None, img_scale=(1., 1.)):
        ymin, xmin, ymax, xmax = nparray
        bbox = BBox(xmin, ymin, xmax, ymax)
        if input_size is not None:
            width, height = input_size
            img_scale_x, img_scale_y = img_scale
            sx, sy = width / img_scale_x, height / img_scale_y
            bbox = bbox.scale(sx, sy).map(int)
        return bbox

    @property
    def nparray(self):
        return np.array(self)

    @property
    def width(self):
        """The bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """The bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """The bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Indicates whether bounding box is valid or not (boolean).
        A valid bounding box has xmin <= xmax and ymin <= ymax (equivalent
        to width >= 0 and height >= 0).
        """
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        """Scales the bounding box.
        Args:
          sx (float): Scale factor for the x-axis.
          sy (float): Scale factor for the y-axis.
        Returns:
          A :obj:`BBox` object with the rescaled dimensions.
        """
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)

    def translate(self, dx, dy):
        """Translates the bounding box position.
        Args:
          dx (int): Number of pixels to move the box on the x-axis.
          dy (int): Number of pixels to move the box on the y-axis.
        Returns:
          A :obj:`BBox` object at the new position.
        """
        return BBox(xmin=dx + self.xmin,
                    ymin=dy + self.ymin,
                    xmax=dx + self.xmax,
                    ymax=dy + self.ymax)

    def map(self, f):
        """Maps all box coordinates to a new position using a given function.
        Args:
          f: A function that takes a single coordinate and returns a new one.
        Returns:
          A :obj:`BBox` with the new coordinates.
        """
        return BBox(xmin=f(self.xmin),
                    ymin=f(self.ymin),
                    xmax=f(self.xmax),
                    ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        """Gets a box representing the intersection between two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          A :obj:`BBox` representing the area where the two boxes intersect
          (may be an invalid box, check with :func:`valid`).
        """
        return BBox(xmin=max(a.xmin, b.xmin),
                    ymin=max(a.ymin, b.ymin),
                    xmax=min(a.xmax, b.xmax),
                    ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        """Gets a box representing the union of two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          A :obj:`BBox` representing the unified area of the two boxes
          (always a valid box).
        """
        return BBox(xmin=min(a.xmin, b.xmin),
                    ymin=min(a.ymin, b.ymin),
                    xmax=max(a.xmax, b.xmax),
                    ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        """Gets the intersection-over-union value for two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          The intersection-over-union value: 1.0 meaning the two boxes are
          perfectly aligned, 0 if not overlapping at all (invalid intersection).
        """
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)


class DetectionRawOutput(collections.namedtuple('DetectionRawOutput', ['boxes', 'class_ids', 'scores', 'count'])):
    """Represents the raw output tensors of the interpreter.
        .. py:attribute:: boxes
            Array containing raw values for all boxes that outcome from the detection
        .. py:attribute:: class_ids
            Array containing raw values for all class ids that outcome from the detection
        .. py:attribute:: scores
            Array containing raw values for all scores that outcome from the detection
        .. py:attribute:: count
            Integer value representing the number of objects that outcome from the detection
    """
    __slots__ = ()

    def save_to_file(self, filename):
        save_tensors_to_file(self._asdict(), filename)

    @staticmethod
    def from_data(data):
        return DetectionRawOutput(
            boxes=data['boxes'],
            class_ids=data['class_ids'],
            scores=data['scores'],
            count=data['count'])

    @staticmethod
    def from_file(filename):
        data = load_tensors_from_file(filename)
        return DetectionRawOutput.from_data(data)

    def objs_from_data(data, threshold=-float('inf')):
        det_out = data.get('detection_output')
        img_scale = data.get('input_image_scale')
        input_size = data.get('model_input_size')
        if type(det_out) is np.ndarray:
            objs = [Object.from_nparray(obj_data, input_size, img_scale) for obj_data in det_out]
            return list(filter(lambda o: o.score >= threshold, objs))
        elif type(det_out) is dict:
            det_raw_out = DetectionRawOutput.from_data(det_out)
            return det_raw_out.get_objects(input_size, img_scale, threshold)

    @staticmethod
    def objs_from_file(filename, threshold=-float('inf')):
        data = load_tensors_from_file(filename)
        return DetectionRawOutput.objs_from_data(data, threshold)

    def get_objects(self, input_size, img_scale=(1., 1.), threshold=-float('inf'), nobjs=None, nparray=False):
        count = nobjs if not nobjs is None else self.count
        width, height = input_size
        img_scale_x, img_scale_y = img_scale
        sx, sy = width / img_scale_x, height / img_scale_y

        def make_object(i):
            if nparray:
                return np.concatenate(([int(self.class_ids[i]), self.scores[i]], self.boxes[i]))
            else:
                ymin, xmin, ymax, xmax = self.boxes[i]
                return Object(
                    id=int(self.class_ids[i]),
                    score=self.scores[i],
                    bbox=BBox(xmin, ymin, xmax, ymax).scale(sx, sy).map(int))

        objs = [make_object(i) for i in range(count) if self.scores[i] >= threshold]
        return np.array(objs, dtype=np.float32) if nparray else objs


def get_detection_raw_output(interpreter):
    return DetectionRawOutput(
        boxes=output_tensor(interpreter, 0)[0],
        class_ids=output_tensor(interpreter, 1)[0],
        scores=output_tensor(interpreter, 2)[0],
        count=int(output_tensor(interpreter, 3)[0]))


def get_objects(interpreter, img_scale=(1., 1.), threshold=-float('inf'), nobjs=None, nparray=False):
    input_size = get_input_size(interpreter)
    return get_detection_raw_output(interpreter).get_objects(input_size, img_scale, threshold, nobjs, nparray)
