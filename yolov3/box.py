# All the classes and functions related to bounding box.

import numpy as np
from scipy.special import expit
import cv2


class BoundBox:
    """
    BoundBox is a predicted bounding box of an object.
    The fields include the position, objectness and class of the bounding box.
    """

    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        # Position of the bounding box.
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        # If the bounding box contains an object.
        self.objness = objness

        # The predicted probability of each class of the object.
        self.classes = classes

        # The predicted label of the object (the class with the highest probability).
        self.label = -1
        # The probability of the predicted label.
        self.score = -1

    def get_label(self):
        """ Return the label of the bounding box. """
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        """ Get the score (probability) of the predicted label. """
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def interval_overlap(interval_a, interval_b):
    """ Return the overlap of two intervals within one dimension. """
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    """
    Compute the IoU (Intersection over Union) of two bounding box.
    This metric is used to determine how much two bounding boxes overlap.
    """
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def sigmoid(x):
    return expit(x)


def softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def do_nms(boxes, nms_thresh):
    """
    Do the non-maximum suppression.
    Select only the bounding box with the highest score among the boxes predicting the same object.
    :param boxes: The predicted bounding boxes for one image.
    :param nms_thresh: The IoU threshold to determine whether two bounding boxes corresponding to the same object.
    :return:
    """

    # Get the number of classes
    if len(boxes) > 0:
        num_class = len(boxes[0].classes)
    else:
        return

    for c in range(num_class):
        # For each class, sort the probability of each bounding box
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        # Starting from the bounding box with higher probability,
        # find all the remaining boxes with IoU higher than threshold with the current box
        # (which means they correspond to the same object),
        # and remove these boxes.
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    """
    Decode the output of the network to bounding boxes.
    The output of one bounding box is an 1-D array, we need to decode it to the attributes of the bounding box.
    :param netout: The output of the network for a single image.
    :param anchors: The pre-defined sizes of the anchor boxes.
    :param obj_thresh: The threshold to determine whether the bounding box contains an object.
    :param net_h: The pre-defined input height to the network.
    :param net_w: The pre-defined input width to the network.
    :return: A list of bounding boxes.
    """
    grid_h, grid_w = netout.shape[:2]
    num_box = 3
    netout = netout.reshape((grid_h, grid_w, num_box, -1))
    num_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = sigmoid(netout[..., :2])
    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(num_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if objectness <= obj_thresh:
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[row, col, b, 5:]

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)

            boxes.append(box)

    return boxes


def preprocess_input(image, net_h, net_w):
    """
    Convert the image with any size to the pre-defined network input size.
    :param image: The input image.
    :param net_h: The pre-defined height of the input to the network.
    :param net_w: The pre-defined width of the input to the network.
    :return: The converted image with (net_h, net_w) size.
    """
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    """
    Correct the sizes of the bounding boxes for the shape of the image.
    Bounding boxes will be stretched back into the shape of the original image.
    Will allow plotting the original image and draw the bounding boxes, hopefully detecting real objects.
    :param boxes: The predicted bounding boxes for one image.
    :param image_h: The height of real image.
    :param image_w: The width of real image.
    :param net_h: The height of input to the network.
    :param net_w: The width of input to the network.
    :return:
    """
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def compute_overlap(a, b):
    """
    Compute the IoU overlap between each pair of predicted boxes and ground truth boxes.
    :param a: N predicted boxes in (N, 4) ndarray of float.
    :param b: K ground truth boxes in (K, 4) ndarray of float.
    :return: The IoU overlap between each pair of predicted boxes and ground truth boxes
        in (N, K) ndarray of float.
    """

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
