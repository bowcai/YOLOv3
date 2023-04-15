import warnings
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

    checkpoint_callback = CustomModelCheckpoint(
        model_to_save=infer_model,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    train_model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator) * train_times_per_epoch,
        epochs=num_epochs,
        verbose=2,
        callbacks=[early_stop_callback, reduce_on_plateau, checkpoint_callback],
        workers=4,
        max_queue_size=8
    )


class CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self, model_to_save, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)

        super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)
