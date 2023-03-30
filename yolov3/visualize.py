# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# import cv2


def draw_boxes(filename, v_boxes, v_labels, v_scores):
    """ Draw the predicted bounding boxes in an image with Matplotlib. """
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        plt.text(x1, y1, label, color='red')
    # show the plot
    plt.show()


# def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
#     for box in boxes:
#         label_str = ''
#         label = -1
#
#         for i in range(len(labels)):
#             if box.classes[i] > obj_thresh:
#                 if label_str != '': label_str += ', '
#                 label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
#                 label = i
#             if not quiet: print(label_str)
#
#         if label >= 0:
#             text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
#             width, height = text_size[0][0], text_size[0][1]
#             region = np.array([[box.xmin - 3, box.ymin],
#                                [box.xmin - 3, box.ymin - height - 26],
#                                [box.xmin + width + 13, box.ymin - height - 26],
#                                [box.xmin + width + 13, box.ymin]], dtype='int32')
#
#             cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(label),
#                           thickness=5)
#             cv2.fillPoly(img=image, pts=[region], color=get_color(label))
#             cv2.putText(img=image,
#                         text=label_str,
#                         org=(box.xmin + 13, box.ymin - 13),
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                         fontScale=1e-3 * image.shape[0],
#                         color=(0, 0, 0),
#                         thickness=2)
#
#     return image
