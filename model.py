# model.py
import tensorflow as tf
from config import Config

def create_model(dropout_rate=None):
    if dropout_rate is None:
        dropout_rate = Config.DROPOUT_RATE
        
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(*Config.IMAGE_SIZE, 3)
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    return model