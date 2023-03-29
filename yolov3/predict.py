import numpy as np
from .box import preprocess_input, decode_netout, correct_yolo_boxes, do_nms


def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    """
    Predict the YOLO bounding boxes for a batch of images.
    :param model: The trained YOLO model.
    :param images: A batch of image with same size.
    :param net_h: The pre-defined input height to the network.
    :param net_w: The pre-defined input width to the network.
    :param anchors: The pre-defined sizes of the anchor boxes.
    :param obj_thresh: The threshold to determine whether the bounding box contains an object.
    :param nms_thresh: The IoU threshold to determine whether two bounding boxes corresponding to the same object.
    :return: The bounding boxes of the batch of images.
    """
    image_h, image_w, _ = images[0].shape
    num_images = len(images)
    batch_input = np.zeros((num_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(num_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

    # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    batch_boxes = []

    for i in range(num_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2 - j) * 6:(3 - j) * 6]  # config['model']['anchors']
            boxes.append(decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w))

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        batch_boxes.append(boxes)

    return batch_boxes
