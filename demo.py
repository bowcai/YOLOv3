import os

from yolov3.load_data import create_training_instances, BatchGenerator
from yolov3.load_weight import load_keras_model, save_keras_model, load_pretrained_weight
from yolov3.model import create_yolov3_model
from yolov3.train import fit_model
from yolov3.evaluate import evaluate

# The path of training and validation set.
train_img_folder = ''
train_annot_folder = ''
train_cache_dir = ''
valid_img_folder = None
valid_annot_folder = None
valid_cache_dir = None

# The labels of object classes.
labels = []

# Batch size.
batch_size = 16

# Warmup batches.
warmup_batches = 3

# If the IoU of predicted bounding box with any ground truth box is higher than the threshold,
# ignore the loss of the predicted box.
ignore_thresh = 0.5

grid_scales = [1, 1, 1]
obj_scale = 5
noobj_scale = 1
xywh_scale = 1
class_scale = 1

# Minimum and maximum input size.
min_input_size = 288
max_input_size = 448

# The pre-defined anchor sizes.
anchors = [55, 69, 75, 234, 133, 240, 136, 129, 142, 363, 203, 290, 228, 184, 285, 359, 341, 260]

# Ratio between network input's size and network output's size, 32 for YOLOv3.
downsample = 32

# The path of pre-trained YOLOv3 weights.
pretrained_weight_path = './weights/yolov3.weights'

# The path of trained Keras model.
keras_model_path = './weights/trained_model.h5'

if __name__ == '__main__':
    # Create training and validation dataset
    train_inst, valid_inst, labels, max_box_per_image = create_training_instances(
        train_img_folder,
        train_annot_folder,
        train_cache_dir,
        valid_img_folder,
        valid_annot_folder,
        valid_cache_dir,
        labels
    )

    # Use the training/validation instances to build the mini-batch generators.
    train_generator = BatchGenerator(
        instances=train_inst,
        anchors=anchors,
        labels=labels,
        downsample=downsample,
        max_box_per_image=max_box_per_image,
        batch_size=batch_size,
        min_net_size=min_input_size,
        max_net_size=max_input_size,
        shuffle=True,
        jitter=0.3,
        norm=True
    )

    valid_generator = BatchGenerator(
        instances=valid_inst,
        anchors=anchors,
        labels=labels,
        downsample=downsample,
        max_box_per_image=max_box_per_image,
        batch_size=batch_size,
        min_net_size=min_input_size,
        max_net_size=max_input_size,
        shuffle=True,
        jitter=0.0,
        norm=True
    )

    # If trained model exists, load the model. Otherwise, create a new model.
    if os.path.isfile(keras_model_path):
        train_model = load_keras_model(keras_model_path)

    else:
        train_model, infer_model = create_yolov3_model(
            nb_class=len(labels),
            anchors=anchors,
            max_box_per_image=max_box_per_image,
            max_grid=[max_input_size, max_input_size],
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            ignore_thresh=ignore_thresh,
            grid_scales=grid_scales,
            obj_scale=obj_scale,
            noobj_scale=noobj_scale,
            xywh_scale=xywh_scale,
            class_scale=class_scale,
        )

        # Load the pre-trained weights into the training model.
        load_pretrained_weight(train_model, pretrained_weight_path, keras_model_path)

    # Fit the model.
    fit_model(train_model, train_generator, valid_generator)

    # Save the model.
    save_keras_model(train_model, keras_model_path)

    # Load the model as the inferring model.
    infer_model = load_keras_model(keras_model_path)

    # Evaluation.
    # Compute mAP for all the classes.
    average_precisions = evaluate(infer_model, valid_generator)

    # Print the score.
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
