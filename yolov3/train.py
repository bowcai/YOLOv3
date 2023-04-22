from .model import dummy_loss
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam


def fit_model(
        train_model,
        infer_model,
        train_generator,
        valid_generator,
        saved_weights_name,
        lr=0.001,
        num_epochs=10,
        train_times_per_epoch=1,
):
    """
    Train the model with training set and validation set.
    :param train_model: The model used to train the parameters.
    :param infer_model: The model used to infer.
    :param train_generator: The batch generator of training set.
    :param valid_generator: The batch generator of validation set.
    :param saved_weights_name: The path of saved weights.
    :param lr: Initial learning rate.
    :param num_epochs: Number of epochs.
    :param train_times_per_epoch: Number of training times per epoch.
    :return:
    """
    optimizer = Adam(learning_rate=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    early_stop_callback = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=7,
        mode='min',
        verbose=1
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )

    train_model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator) * train_times_per_epoch,
        epochs=num_epochs,
        verbose=2,
        callbacks=[early_stop_callback, reduce_on_plateau],
        workers=4,
        max_queue_size=8
    )
