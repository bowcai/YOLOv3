import numpy as np
import cv2
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


# def draw_boxes(filename, v_boxes, v_labels, v_scores):
#     """ Draw the predicted bounding boxes in an image with Matplotlib. """
#     # load the image
#     data = plt.imread(filename)
#     # plot the image
#     plt.imshow(data)
#     # get the context for drawing boxes
#     ax = plt.gca()
#     # plot each box
#     for i in range(len(v_boxes)):
#         box = v_boxes[i]
#         # get coordinates
#         y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
#         # calculate width and height of the box
#         width, height = x2 - x1, y2 - y1
#         # create the shape
#         rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#         # draw the box
#         ax.add_patch(rect)
#         # draw text and score in top left corner
#         label = "%s (%.3f)" % (v_labels[i], v_scores[i])
#         plt.text(x1, y1, label, color='red')
#     # show the plot
#     plt.show()


colors = [
    [31  , 0   , 255] ,
    [0   , 159 , 255] ,
    [255 , 95  , 0]   ,
    [255 , 19  , 0]   ,
    [255 , 0   , 0]   ,
    [255 , 38  , 0]   ,
    [0   , 255 , 25]  ,
    [255 , 0   , 133] ,
    [255 , 172 , 0]   ,
    [108 , 0   , 255] ,
    [0   , 82  , 255] ,
    [0   , 255 , 6]   ,
    [255 , 0   , 152] ,
    [223 , 0   , 255] ,
    [12  , 0   , 255] ,
    [0   , 255 , 178] ,
    [108 , 255 , 0]   ,
    [184 , 0   , 255] ,
    [255 , 0   , 76]  ,
    [146 , 255 , 0]   ,
    [51  , 0   , 255] ,
    [0   , 197 , 255] ,
    [255 , 248 , 0]   ,
    [255 , 0   , 19]  ,
    [255 , 0   , 38]  ,
    [89  , 255 , 0]   ,
    [127 , 255 , 0]   ,
    [255 , 153 , 0]   ,
    [0   , 255 , 255] ,
    [0   , 255 , 216] ,
    [0   , 255 , 121] ,
    [255 , 0   , 248] ,
    [70  , 0   , 255] ,
    [0   , 255 , 159] ,
    [0   , 216 , 255] ,
    [0   , 6   , 255] ,
    [0   , 63  , 255] ,
    [31  , 255 , 0]   ,
    [255 , 57  , 0]   ,
    [255 , 0   , 210] ,
    [0   , 255 , 102] ,
    [242 , 255 , 0]   ,
    [255 , 191 , 0]   ,
    [0   , 255 , 63]  ,
    [255 , 0   , 95]  ,
    [146 , 0   , 255] ,
    [184 , 255 , 0]   ,
    [255 , 114 , 0]   ,
    [0   , 255 , 235] ,
    [255 , 229 , 0]   ,
    [0   , 178 , 255] ,
    [255 , 0   , 114] ,
    [255 , 0   , 57]  ,
    [0   , 140 , 255] ,
    [0   , 121 , 255] ,
    [12  , 255 , 0]   ,
    [255 , 210 , 0]   ,
    [0   , 255 , 44]  ,
    [165 , 255 , 0]   ,
    [0   , 25  , 255] ,
    [0   , 255 , 140] ,
    [0   , 101 , 255] ,
    [0   , 255 , 82]  ,
    [223 , 255 , 0]   ,
    [242 , 0   , 255] ,
    [89  , 0   , 255] ,
    [165 , 0   , 255] ,
    [70  , 255 , 0]   ,
    [255 , 0   , 172] ,
    [255 , 76  , 0]   ,
    [203 , 255 , 0]   ,
    [204 , 0   , 255] ,
    [255 , 0   , 229] ,
    [255 , 133 , 0]   ,
    [127 , 0   , 255] ,
    [0   , 235 , 255] ,
    [0   , 255 , 197] ,
    [255 , 0   , 191] ,
    [0   , 44  , 255] ,
    [50  , 255 , 0]
]


def get_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    code originally from https://github.com/fizyr/keras-retinanet/
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
    """
    if label < len(colors):
        return colors[label]
    else:
        print('Label {} has no color, returning default.'.format(label))
        return [0, 255, 0]


def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    """ Draw the predicted bounding boxes in an image with OpenCV. """
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '':
                    label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
            if not quiet:
                print(label_str)

        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin - 3, box.ymin],
                               [box.xmin - 3, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin]], dtype='int32')

            cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(label),
                          thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin + 13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(0, 0, 0),
                        thickness=2)

    return image
