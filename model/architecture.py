"""
NeuroLock AI — Model Architecture
Mini-Xception CNN optimized for Facial Expression Recognition
Based on: "Real-time Convolutional Neural Networks for Emotion and Gender Classification"
          Arriaga et al., 2017 — best known lightweight model for FER2013
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization,
    Activation, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Add, Flatten
)
from tensorflow.keras.regularizers import l2
import tensorflow as tf


EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
NUM_CLASSES     = len(EMOTION_LABELS)
INPUT_SHAPE     = (48, 48, 1)


def residual_block(x, num_filters: int, l2_reg: float = 0.01):
    """Depthwise-separable residual block."""
    residual = Conv2D(num_filters, 1, strides=2, padding='same',
                      use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(num_filters, 3, padding='same',
                        use_bias=False, depthwise_regularizer=l2(l2_reg),
                        pointwise_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(num_filters, 3, padding='same',
                        use_bias=False, depthwise_regularizer=l2(l2_reg),
                        pointwise_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    x = Add()([x, residual])
    return x


def build_mini_xception(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int   = NUM_CLASSES,
    l2_reg: float      = 0.01,
) -> Model:
    """
    Mini-Xception architecture.
    ~0.6M parameters — fast inference, high accuracy on FER2013.
    Expected accuracy: ~65–67% on FER2013 test set.
    """
    inputs = Input(shape=input_shape, name='input')

    # ── Stem ──────────────────────────────────────────────
    x = Conv2D(8, 3, strides=1, padding='same', use_bias=False,
               kernel_regularizer=l2(l2_reg), name='stem_conv')(inputs)
    x = BatchNormalization(name='stem_bn')(x)
    x = Activation('relu', name='stem_act')(x)

    x = Conv2D(8, 3, strides=1, padding='same', use_bias=False,
               kernel_regularizer=l2(l2_reg), name='stem_conv2')(x)
    x = BatchNormalization(name='stem_bn2')(x)
    x = Activation('relu', name='stem_act2')(x)

    # ── Residual Blocks ───────────────────────────────────
    for i, num_filters in enumerate([16, 32, 64, 128], start=1):
        x = residual_block(x, num_filters, l2_reg)

    # ── Head ──────────────────────────────────────────────
    x = Conv2D(num_classes, 3, padding='same',
               kernel_regularizer=l2(l2_reg), name='head_conv')(x)
    x = GlobalAveragePooling2D(name='gap')(x)
    outputs = Activation('softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='NeuroLock_MiniXception')
    return model


def build_deep_cnn(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int   = NUM_CLASSES,
) -> Model:
    """
    Deeper plain CNN alternative with more capacity.
    Useful when Xception underfits on custom data splits.
    """
    inputs = Input(shape=input_shape, name='input')

    def conv_block(x, filters, kernel=3, pool=True):
        x = Conv2D(filters, kernel, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if pool:
            x = MaxPooling2D(2)(x)
            x = Dropout(0.25)(x)
        return x

    x = conv_block(inputs,  32)
    x = conv_block(x,       64)
    x = conv_block(x,       128)
    x = conv_block(x,       256, pool=False)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='NeuroLock_DeepCNN')
    return model


def model_summary(model: Model) -> None:
    model.summary()
    total_params = model.count_params()
    print(f"\n  Total parameters : {total_params:,}")
    print(f"  Model size (est) : ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)\n")


if __name__ == '__main__':
    m = build_mini_xception()
    model_summary(m)
