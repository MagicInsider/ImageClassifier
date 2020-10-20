import os
import sys
import time
from faker import Faker
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from clr import CyclicLR
from models import ResNet50, ResNet34, ResNetXX
from models_seq import ResNet50_seq, DG
from quantize import quantize_model


def root_path():
    return os.path.dirname(sys.argv[0])


def make_dir(path):
    try:
        os.umask(0)
        os.mkdir(path, mode=0o777)
    except FileExistsError:
        pass
    return path


def get_readable_runtime(tic_, toc_):
    """
    :parameters
    tic -- start time, system time
    toc -- finish time, system time
    :returns
    runtime - string in format 'MM min SS sec'
    """
    runtime_raw = toc_ - tic_
    runtime = str(int(runtime_raw // 60)) + ' min ' + str(int(runtime_raw % 60)) + ' sec'
    return runtime


def set_data(input_dims_, dataset_path_, val_split_=.0,
             train_batch_size_=1, val_batch_size_=1, use_aug_=False, save_aug_=False):

    classes_ = examples_ = 0
    for _, dirnames, filenames in os.walk(dataset_path_):
        classes_ += len(dirnames)
        examples_ += len(filenames)
    train_batches_per_epoch_ = int(np.ceil(np.floor((examples_ * (1 - val_split_))) / train_batch_size_))

    save_to_dir = train_aug_prefix = val_aug_prefix = None
    if use_aug_ and save_aug_:
        save_to_dir = make_dir(os.path.join(root_path(), 'data_temp'))
        train_aug_prefix = 'train_'
        val_aug_prefix = 'val_'

    if use_aug_:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_epsilon=1e-6,
            zca_whitening=False,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(.7, 1.3),
            shear_range=10,
            zoom_range=[.95, 1.2],
            channel_shift_range=.4,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=False,
            rescale=1. / 255,
            validation_split=val_split_
        )

    else:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=val_split_
        )

    train_flow = datagen.flow_from_directory(
        directory=dataset_path_,
        target_size=input_dims_,
        batch_size=train_batch_size_,
        class_mode='categorical',
        shuffle=True,
        save_to_dir=save_to_dir,
        save_prefix=train_aug_prefix,
        subset='training',
        interpolation='bilinear'
    )

    val_flow = datagen.flow_from_directory(
        directory=dataset_path_,
        target_size=input_dims_,
        batch_size=val_batch_size_,
        class_mode='categorical',
        shuffle=True,
        save_to_dir=save_to_dir,
        save_prefix=val_aug_prefix,
        subset='validation',
        interpolation='bilinear'
    )
    return train_flow, val_flow, classes_, train_batches_per_epoch_


def set_model(input_dims_, model_type_, classes_):
    input_shape = input_dims_ + (3,)
    if model_type_ == 'ResNet50':
        model_ = ResNet50(input_shape=input_shape, classes=classes_, name='ResNet50')
    elif model_type_ == 'ResNet50_seq':
        model_ = ResNet50_seq(input_shape=input_shape, classes=classes_, name='ResNet50_seq')
    elif model_type_ == 'ResNet34':
        model_ = ResNet34(input_shape=input_shape, classes=classes_, name='ResNet34')
    elif model_type_ == 'ResNetXX':
        model_ = ResNetXX(input_shape=input_shape, classes=classes_, name='ResNetXX')
    elif model_type_ == 'DG':
        model_ = DG(input_shape=input_shape, classes=classes_, name='DG')

    model_name_ = '_'.join([model_.name, 'x'.join(map(str, input_dims_)), '_'.join(Faker().name().split(' '))])
    return model_, model_name_


def set_optimizer(type_, lr=.001, beta1=.9, beta2=.999, epsilon=1e-07, rho=.9, momentum=.0,
                  amsgrad=False, nesterov=False):
    if type_ == 'Adam':
        optimizer_ = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name="Adam"
         )
    elif type_ == 'RMS':
        optimizer_ = tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=False,
            name='RMSprop'
        )
    elif type_ == 'SGD':
        optimizer_ = tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=nesterov,
            name="SGD",
        )
    return optimizer_


def set_paths_and_callbacks(callbacks_list_, val_split_):
    tensorboard_path = make_dir(os.path.join(make_dir(os.path.join(root_path(), 'tensorboard')), model_name))
    model_path = make_dir(os.path.join(make_dir(os.path.join(root_path(), 'models')), model_name))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
    callbacks_list_.append(tensorboard_callback)

    file_writer = tf.summary.create_file_writer(logdir=tensorboard_path + "/learning_rate")
    file_writer.set_as_default()

    categorical_acc_best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_path, 'weights.cat_acc_best.hdf5'),
        save_weights_only=True,
        monitor='categorical_accuracy',
        mode='max',
        save_best_only=True
    )
    callbacks_list_.append(categorical_acc_best_checkpoint)

    if val_split_ != 0:
        val_categorical_acc_best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, 'weights.ep-{epoch:02d}_val_cat_acc-{val_categorical_accuracy:.2f}.hdf5'),
            save_weights_only=True,
            monitor='val_categorical_accuracy',
            mode='max',
            save_best_only=True)
        callbacks_list_.append(val_categorical_acc_best_checkpoint)

    return callbacks_list_, model_path


if __name__ == '__main__':
    model_type = 'DG'   # ResNet50, ResNet50_seq, ResNet34, ResNetXX, DG
    input_dims = (64, 64)
    dataset_path = os.path.join(root_path(), 'data_raw')
    val_split = 0
    train_batch_size = 8
    val_batch_size = 8
    use_aug = True
    save_aug = False

    optimizer_type = 'SGD'   # Adam, RMS, SGD
    top_lr = .005
    bottom_lr = .0001
    optimizer = set_optimizer(type_=optimizer_type, lr=top_lr, momentum=.9, nesterov=True)

    epochs = 62

    lr_mode = 'cyclic'   # test, cyclic, flat
    epochs_per_clr = 2
    clr_mode = 'warm-restart'  # triangular, triangular-gamma, exp-omega, abs-cosine, abs-sine, 'warm-restart'
    clr_profile_fn = None
    clr_scale_fn = None   # lambda x: np.sin((x / (epochs / epochs_per_clr)) * np.pi)
    clr_gamma = None
    clr_omega = None
    clr_start_mode = None
    clr_cycle_mode = None
    clr_moderator = 2

    quantize = False

    metrics = [
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            ]

    init_epoch = 0
    callbacks = []

    train_source, val_source, classes, batches_per_epoch =\
        set_data(input_dims, dataset_path, val_split, train_batch_size, val_batch_size, use_aug, save_aug)

    model, model_name = set_model(input_dims, model_type, classes)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics
    )

    if lr_mode == 'test':
        callbacks.append(CyclicLR(bottom_lr=bottom_lr, top_lr=top_lr, iter_per_cycle=epochs * batches_per_epoch,
                                  mode='triangular', start_mode='bottom-start', cycle_mode='one-way'))

    elif lr_mode == 'cyclic':
        callbacks.append(CyclicLR(bottom_lr=bottom_lr, top_lr=top_lr, iter_per_cycle=epochs_per_clr * batches_per_epoch,
                                  mode=clr_mode, profile_fn=clr_profile_fn, scale_fn=clr_scale_fn,
                                  gamma=clr_gamma, omega=clr_omega, start_mode=clr_start_mode, cycle_mode=clr_cycle_mode,
                                  moderator=clr_moderator))

    callbacks, model_path = set_paths_and_callbacks(callbacks, val_split)

    print(f"\n\nSTARTING training model {model_name} for {epochs} epochs\n")
    tic = time.time()
    epochs += init_epoch

    model.fit(
        train_source,
        initial_epoch=init_epoch,
        epochs=epochs,
        validation_data=val_source,
        verbose=1,
        callbacks=callbacks
    )

    toc = time.time()
    print(f'\n\nFINISHED training of {model_name} in {get_readable_runtime(tic, toc)}\n\n')

    if quantize:
        quantize_model(model, model_path)

    sys.exit(0)
