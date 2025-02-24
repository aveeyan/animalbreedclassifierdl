# config.py
import os

class Config:
    # Dataset paths
    DATA_ROOT = "dataset-iiit-pet"
    IMAGES_DIR = os.path.join(DATA_ROOT, "images")
    ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
    TRAIN_LIST = os.path.join(ANNOTATIONS_DIR, "trainval.txt")
    TEST_LIST = os.path.join(ANNOTATIONS_DIR, "test.txt")
    
    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    
    # Training settings
    VALIDATION_SPLIT = 0.2
    USE_AUGMENTATION = True
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    USE_LR_REDUCTION = True
    LR_REDUCTION_PATIENCE = 5
    LR_REDUCTION_FACTOR = 0.2
    
    # Classes
    NUM_CLASSES = 37
    
    # Paths
    MODEL_SAVE_PATH = "models/animal_breed_classifier.h5"
    CLASS_NAMES_PATH = "models/class_names.json"
