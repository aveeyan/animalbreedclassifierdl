# inference.py
import tensorflow as tf
import numpy as np
from PIL import Image
from config import Config

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize(Config.IMAGE_SIZE)
    img_array = np.array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict(model, image_path, class_names):
    # Preprocess the image
    processed_image = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {
            'class': class_names[idx],
            'confidence': float(predictions[0][idx])
        }
        for idx in top_3_idx
    ]
    
    return {
        'top_prediction': {
            'class': class_names[predicted_class_idx],
            'confidence': float(confidence)
        },
        'top_3_predictions': top_3_predictions
    }