# Keras custom pose model
#
# Primary NN architecture based on Mobilenetv2
#
# Final layers based on Microsoft code here:
# https://github.com/microsoft/human-pose-estimation.pytorch
#
# Transfer learning approach based on:
# https://www.tensorflow.org/tutorials/images/transfer_learning

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal, Constant

BN_MOMENTUM = 0.1       # For final deconv layers only


def PoseMobileNetV2(cfg, alpha):

    # Setup base Mobilenet V2 NN
    IMG_SHAPE = (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1], 3)
    base_model = MobileNetV2(input_shape=IMG_SHAPE,
                             include_top=False,
                             alpha=alpha,
                             weights='imagenet')

    base_model.trainable = False

    # Drop final set of layers to reduce output from 1280 to 320
    # Reduces trainable parameters from 9.5m to 5.2m
    base_model_output = base_model.get_layer('block_16_project_BN').output

    extra = cfg.MODEL.EXTRA
    deconv_with_bias = extra.DECONV_WITH_BIAS

    kernel_initializer = RandomNormal(stddev=0.001)
    bias_initializer = Constant(0.)

    # Each layer created manually due to nuances between original Pytorch code and tf.keras version
    # Deconv layer 0
    planes = extra.NUM_DECONV_FILTERS[0]
    deconv_0 = Conv2DTranspose(
        filters=planes,
        kernel_size=4,
        strides=2,
        padding='same',
        use_bias=deconv_with_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(base_model_output)
    batchnorm_0 = BatchNormalization(momentum=BN_MOMENTUM)(deconv_0)
    relu_0 = ReLU()(batchnorm_0)

    # Deconv layer 1
    planes = extra.NUM_DECONV_FILTERS[1]
    deconv_1 = Conv2DTranspose(
        filters=planes,
        kernel_size=4,
        strides=2,
        padding='same',
        use_bias=deconv_with_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(relu_0)
    batchnorm_1 = BatchNormalization(momentum=BN_MOMENTUM)(deconv_1)
    relu_1 = ReLU()(batchnorm_1)

    # Deconv layer 2
    planes = extra.NUM_DECONV_FILTERS[2]
    deconv_2 = Conv2DTranspose(
        filters=planes,
        kernel_size=4,
        strides=2,
        padding='same',
        use_bias=deconv_with_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(relu_1)
    batchnorm_2 = BatchNormalization(momentum=BN_MOMENTUM)(deconv_2)
    relu_2 = ReLU()(batchnorm_2)

    # Final layer
    padding = 'same' if extra.FINAL_CONV_KERNEL == 3 else 'valid'
    final_layer = Conv2D(
        filters=cfg.MODEL.NUM_JOINTS,
        kernel_size=extra.FINAL_CONV_KERNEL,
        strides=1,
        padding=padding,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(relu_2)

    # Build complete model
    model = Model(inputs=base_model.inputs, outputs=final_layer, name='pose_mobilenetv2_%0.2f' % alpha)

    return model


def get_pose_net(cfg, is_train, alpha=1.0):
    """
    get_pose_net() provides same interface as resnet50 for NN config
    """

    if cfg.MODEL.INIT_WEIGHTS:
        # Load pre-trained model
        model = load_model(cfg.MODEL.PRETRAINED)
    else:
        # Load pretrained mobilenet)_v2 network and setup deconv layers
        model = PoseMobileNetV2(cfg, alpha=alpha)

    return model