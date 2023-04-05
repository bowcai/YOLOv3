from .model import dummy_loss
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


def fit_model(
        model,
        train_generator,
        valid_generator,
        lr=0.001,
        num_epochs=10,
        train_times_per_epoch=1
):
    """
    Train the model with training set and validation set.
    :param model: The model to be trained.
    :param train_generator: The batch generator of training set.
    :param valid_generator: The batch generator of validation set.
    :param lr: Initial learning rate.
    :param num_epochs: Number of epochs.
    :param train_times_per_epoch: Number of training times per epoch.
    :return:
    """
    optimizer = Adam(lr=lr, clipnorm=0.001)
    model.compile(loss=dummy_loss, optimizer=optimizer)

    early_stop_callback = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=7,
        mode='min',
        verbose=1
    )

    model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator) * train_times_per_epoch,
        epochs=num_epochs,
        verbose=2,
        callbacks=early_stop_callback,
        workers=4,
        max_queue_size=8
    )
