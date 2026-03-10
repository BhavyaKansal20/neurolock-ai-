"""
NeuroLock AI v2 — MobileNetV2 Transfer Learning Model (FIXED)
==============================================================
FIXES:
1. Lambda layer hataya (tf.repeat wala) — yeh .h5 load hone se rok raha tha
   Keras 3 mein Lambda layer safe_mode=False bina deserialize nahi hoti
2. Conv2D(3, 1x1) se grayscale to 3ch conversion — fully serializable
3. alpha=0.35 to alpha=1.0 — proper full MobileNetV2 (significantly more powerful)
4. Dense layers 256/128 to 512/256 — better capacity for 7 emotion classes
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, Conv2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from model.architecture import EMOTION_LABELS, NUM_CLASSES


def build_mobilenet_v2(num_classes: int = NUM_CLASSES) -> Model:
    """
    MobileNetV2 fine-tuned for FER.
    Input: 96x96 grayscale
    FIX: Lambda(tf.repeat) ki jagah Conv2D(3, 1, padding='same') use kiya.
         Yeh fully serializable hai — .keras aur .h5 dono mein safely load hota hai.
         Model khud seekhta hai ki gray channel ko 3ch mein kaise map kare.
    """
    inputs = Input(shape=(96, 96, 1), name='input')

    # FIX: Lambda ki jagah 1x1 Conv2D se gray to 3-channel projection
    # use_bias=False — sirf weights, koi bias nahi
    x = Conv2D(3, (1, 1), padding='same', use_bias=False, name='gray_to_3ch')(inputs)

    # MobileNetV2 backbone — alpha=1.0 full model (alpha=0.35 bahut weak tha)
    base = MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    base.trainable = False  # Phase 1: sirf head train karo

    x = base(x, training=False)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='NeuroLock_MobileNetV2')
    return model, base


def unfreeze_top_layers(base_model: Model, num_layers: int = 40):
    """Phase 2: Top N layers unfreeze karo fine-tuning ke liye."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    print(f"  Unfroze top {num_layers} layers of MobileNetV2")


INPUT_SHAPE_MOBILENET = (96, 96, 1)
