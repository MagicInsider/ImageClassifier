from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import glorot_uniform


@tf.keras.utils.register_keras_serializable(package='Custom', name='ResnetIdentityBlock')
class ResnetIdentityBlock(Model, ABC):
    def __init__(self, kernel_size, filters, stage, block):
        super(ResnetIdentityBlock, self).__init__()

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv2a = Conv2D(
            filters=filters1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2a',
            kernel_initializer=glorot_uniform
            )
        self.bn2a = BatchNormalization(
            axis=3,
            name=bn_name_base + '2a'
            )

        self.conv2b = Conv2D(
            filters=filters2,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            name=conv_name_base + '2b',
            kernel_initializer=glorot_uniform
            )

        self.bn2b = BatchNormalization(
            axis=3,
            name=bn_name_base + '2b'
            )

        self.conv2c = Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2c',
            kernel_initializer=glorot_uniform
            )
        self.bn2c = BatchNormalization(
            axis=3,
            name=bn_name_base + '2b'
            )

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

    def get_config(self):
        config = super(ResnetIdentityBlock, self).get_config()
        config.update(
            {
                'conv2a': self.conv2a,
                'bn2a': self.bn2a,
                'conv2b': self.conv2b,
                'bn2b': self.bn2b,
                'conv2c': self.conv2c, 
                }
        )
        return config


@tf.keras.utils.register_keras_serializable(package='Custom', name='ResnetConvolutionalBlock')
class ResnetConvolutionalBlock(Model, ABC):
    def __init__(self, kernel_size, filters, strides, stage, block):
        super(ResnetConvolutionalBlock, self).__init__()

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv2a = Conv2D(
            filters=filters1,
            kernel_size=(1, 1),
            strides=strides,
            padding='valid',
            name=conv_name_base + '2a',
            kernel_initializer=glorot_uniform
            )

        self.bn2a = BatchNormalization(
            axis=3,
            name=bn_name_base + '2a'
            )

        self.conv2b = Conv2D(
            filters=filters2,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            name=conv_name_base + '2b',
            kernel_initializer=glorot_uniform
            )

        self.bn2b = BatchNormalization(
            axis=3,
            name=bn_name_base + '2b'
            )

        self.conv2c = Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2c',
            kernel_initializer=glorot_uniform
            )
        self.bn2c = BatchNormalization(
            axis=3,
            name=bn_name_base + '2b'
            )

        self.conv_shortcut = Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            strides=strides,
            padding='valid',
            name=conv_name_base + 'shortcut_1',
            kernel_initializer=glorot_uniform
            )
        self.bn_shortcut = BatchNormalization(
            axis=3,
            name=bn_name_base + 'shortcut_1'
            )

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)

    def get_config(self):
        config = super(ResnetConvolutionalBlock, self).get_config()
        config.update = (
            {
                'conv2a': self.conv2a, 
                'bn2a': self.bn2a, 
                'conv2b': self.conv2b, 
                'bn2b': self.bn2b, 
                'conv2c': self.conv2c, 
                'conv_shortcut': self.conv_shortcut, 
                'bn_shortcut': self.bn_shortcut, 
                }
            )
        return config


def ResNet50_seq(input_shape, classes, name):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK x 2 -> CONVBLOCK -> IDBLOCK x 3
    -> CONVBLOCK -> IDBLOCK x 5 -> CONVBLOCK -> IDBLOCK x 2 -> AVGPOOL
    -> DROPOUT -> DENSE -> BATCHNORM -> DROPOUT -> DENSE -> SOFTMAX

    Arguments:
    input_shape -- tuple (height_px, width_px, channel), shape of the images of the dataset
    classes -- integer, number of classes
    name -- str, name of the model

    Returns:
    model -- a Keras Sequiential Model instance
    """
    model = Sequential(
        [
            Input(input_shape),
            ZeroPadding2D((3, 3)),
            Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform),
            BatchNormalization(axis=3, name='bn_conv1'),
            Activation('relu'),
            MaxPooling2D((3, 3), strides=(2, 2)),

            ResnetConvolutionalBlock(kernel_size=(3, 3), filters=(64, 64, 256), strides=(1, 1), stage=2, block='a'),
            ResnetIdentityBlock(kernel_size=(3, 3), filters=(64, 64, 256), stage=2, block='b'),
            ResnetIdentityBlock(kernel_size=(3, 3), filters=(64, 64, 256), stage=2, block='c'),

            ResnetConvolutionalBlock(kernel_size=(3, 3), filters=(128, 128, 512), strides=(2, 2), stage=3, block='a'),
            ResnetIdentityBlock(kernel_size=(3, 3), filters=(128, 128, 512), stage=3, block='b'),
            ResnetIdentityBlock(kernel_size=(3, 3), filters=(128, 128, 512), stage=3, block='c'),
            ResnetIdentityBlock(kernel_size=(3, 3), filters=(128, 128, 512), stage=3, block='d'),

            ResnetConvolutionalBlock(kernel_size=(3, 3), filters=(256, 256, 1024), strides=(2, 2), stage=4, block='a'),
            ResnetIdentityBlock(kernel_size=(3, 3),  filters=(256, 256, 1024), stage=4, block='b'),
            ResnetIdentityBlock(kernel_size=(3, 3),  filters=(256, 256, 1024), stage=4, block='c'),
            ResnetIdentityBlock(kernel_size=(3, 3),  filters=(256, 256, 1024), stage=4, block='d'),
            ResnetIdentityBlock(kernel_size=(3, 3),  filters=(256, 256, 1024), stage=4, block='e'),
            ResnetIdentityBlock(kernel_size=(3, 3),  filters=(256, 256, 1024), stage=4, block='f'),

            ResnetConvolutionalBlock(kernel_size=(3, 3), filters=(512, 512, 2048), strides=(2, 2), stage=5, block='a'),
            ResnetIdentityBlock(kernel_size=(3, 3), filters=(512, 512, 2048), stage=5, block='b'),
            ResnetIdentityBlock(kernel_size=(3, 3), filters=(512, 512, 2048), stage=5, block='c'),

            AveragePooling2D(pool_size=(2, 2), name='avg_pool'),
            Flatten(),
            Dropout(0.4, name='dropout_fc_' + str(classes * 8)),
            Dense(classes * 8, name='fc_' + str(classes * 8), kernel_initializer=glorot_uniform),
            BatchNormalization(name='bn_fc' + str(classes)),
            Activation('relu'),
            Dropout(0.25, name='dropout_fc_' + str(classes)),
            Dense(classes, activation='softmax', name='fc_' + str(classes), kernel_initializer=glorot_uniform)
        ],
        name=name
        )

    return model


def DG(input_shape, classes, name):
    """
    Conv2D -> FC
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    model = Sequential(
        [
            Input(input_shape),
            Conv2D(64, (7, 7), strides=(2, 2), name='conv', kernel_initializer=glorot_uniform),
            Flatten(),
            Dense(classes, activation='softmax', name='fc', kernel_initializer=glorot_uniform),
        ],
        name=name
    )

    return model
