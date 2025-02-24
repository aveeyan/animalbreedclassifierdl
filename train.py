# train.py
import tensorflow as tf
import json
from config import Config
from dataset import PetDataset
from model import create_model

def train(dropout_rate=None,
          use_augmentation=None,
          use_early_stopping=None,
          early_stopping_patience=None,
          use_lr_reduction=None,
          lr_reduction_patience=None,
          lr_reduction_factor=None):
    
    # Update configuration with provided parameters
    if dropout_rate is not None:
        Config.DROPOUT_RATE = dropout_rate
    if use_augmentation is not None:
        Config.USE_AUGMENTATION = use_augmentation
    if use_early_stopping is not None:
        Config.USE_EARLY_STOPPING = use_early_stopping
    if early_stopping_patience is not None:
        Config.EARLY_STOPPING_PATIENCE = early_stopping_patience
    if use_lr_reduction is not None:
        Config.USE_LR_REDUCTION = use_lr_reduction
    if lr_reduction_patience is not None:
        Config.LR_REDUCTION_PATIENCE = lr_reduction_patience
    if lr_reduction_factor is not None:
        Config.LR_REDUCTION_FACTOR = lr_reduction_factor
    
    # Create dataset
    pet_dataset = PetDataset()
    train_ds, val_ds = pet_dataset.create_datasets()
    
    # Create and compile model
    model = create_model(dropout_rate=Config.DROPOUT_RATE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare callbacks
    callbacks = []
    
    # Model checkpoint callback (always included)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            Config.MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_accuracy'
        )
    )
    
    # Early stopping callback
    if Config.USE_EARLY_STOPPING:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
        )
    
    # Learning rate reduction callback
    if Config.USE_LR_REDUCTION:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=Config.LR_REDUCTION_FACTOR,
                patience=Config.LR_REDUCTION_PATIENCE,
                min_lr=1e-6
            )
        )
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )
    
    # Save class names
    with open(Config.CLASS_NAMES_PATH, 'w') as f:
        json.dump(pet_dataset.class_names, f)
    
    return model, history, pet_dataset.class_names