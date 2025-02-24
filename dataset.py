import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from config import Config
import json

class PetDataset:
    def __init__(self):
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self._load_dataset()
    
    def _parse_dataset_file(self, file_path):
        paths = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        class_id = int(parts[1]) - 1  # assuming class_id starts at 1
                        image_path = os.path.join(Config.IMAGES_DIR, f"{image_name}.jpg")
                        if os.path.exists(image_path):
                            paths.append(image_path)
                            labels.append(class_id)
        return paths, labels
    
    def _load_dataset(self):
        train_paths, train_labels = self._parse_dataset_file(Config.TRAIN_LIST)
        test_paths, test_labels = self._parse_dataset_file(Config.TEST_LIST)
        
        self.image_paths = train_paths + test_paths
        self.labels = train_labels + test_labels
        
        # Modify the way class names are extracted to handle entire breed names
        self.class_names = sorted(list(set([
            "_".join(os.path.basename(path).split('_')[:-1])  # Join all parts except the last part of the name
            for path in self.image_paths
        ])))
    
    def apply_augmentation(self, image):
        if Config.USE_AUGMENTATION:
            # Apply various augmentations
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.1)
            image = tf.image.random_crop(image, size=[Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3])  # Random crop
            image = tf.image.resize_with_crop_or_pad(image, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])  # Ensure the size is fixed
            return image
        return image
    
    def preprocess_image(self, image_path, augment=True):
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, Config.IMAGE_SIZE)
        
        # Apply augmentation during training
        if augment and Config.USE_AUGMENTATION:
            image = self.apply_augmentation(image)
        
        # Normalize image to [0,1] or using the specific model's preprocessing
        image = tf.keras.applications.resnet50.preprocess_input(image)  # Preprocessing for ResNet50
        return image
    
    def create_datasets(self):
        # Split dataset into training and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.image_paths,
            self.labels,
            test_size=Config.VALIDATION_SPLIT,
            stratify=self.labels,
            random_state=42
        )
        
        # Create tensorflow datasets from the paths and labels
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        
        # Apply transformations to the training dataset
        train_ds = (train_ds
            .map(lambda x, y: (self.preprocess_image(x, augment=True), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(5000)  # Increased shuffle buffer size for better randomness
            .batch(Config.BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))  # Prefetch to improve performance
        
        # Apply transformations to the validation dataset
        val_ds = (val_ds
            .map(lambda x, y: (self.preprocess_image(x, augment=False), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(Config.BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))  # Prefetch to improve performance
        
        return train_ds, val_ds
