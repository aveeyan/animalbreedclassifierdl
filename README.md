# Animal Breed Classification

## Overview
This project is a deep learning-based animal breed classification system. It utilizes a trained convolutional neural network (CNN) to identify the breed of an animal from an input image. The model is trained using TensorFlow/Keras and can classify images into multiple animal breeds.

## Features
- **Deep Learning Model**: Uses a CNN trained on a large dataset of animal images.
- **Image Classification**: Predicts the breed of an animal from an uploaded image.
- **Pretrained Model**: Loads a pre-trained model for fast and accurate predictions.
- **JSON Class Mapping**: Uses a JSON file to map class indices to breed names.
- **Configurable Paths**: Uses `config.py` for flexible path management.
- **Dataset Automation**: Includes a setup script (`setup.sh`) to automatically download and prepare the dataset.

## Project Structure
```
|-- dataset-iiit-pet/  # Cloned Oxford-IIIT Pet Dataset
|-- graphs/            # Stores visualizations and analysis plots
|-- models/            # Stores trained models and checkpoints
|-- config.py          # Configuration settings
|-- dataset.py         # Handles dataset loading and preprocessing
|-- inference.py       # Script for making predictions
|-- model.py           # Defines the model architecture
|-- train.py           # Training script
|-- main.py            # Entry point for running classification
|-- requirements.txt   # Dependencies
|-- setup.sh           # Script to download dataset and install dependencies
|-- README.md          # Project documentation
```

## Setup & Installation
### Prerequisites
Ensure you have Python installed (recommended: Python 3.8+). Install dependencies using:
```bash
pip install -r requirements.txt
```

### Download Dataset
Run the setup script to automatically download and configure the dataset:
```bash
bash setup.sh
```

### Running the Classifier
To classify an image, run:
```bash
python3 main.py
```

## Configuration
All configurable paths (such as model and class mapping file paths) are stored in `config.py`. Modify this file as needed:
```python
class Config:
    BEST_MODEL_PATH = "models/best_model.h5"
    CLASS_NAMES_PATH = "models/class_names.json"
```

## Training the Model
To retrain the model on a new dataset, modify `train.py` and run:
```bash
python train.py
```

## Acknowledgments
This project uses TensorFlow/Keras for deep learning and OpenCV for image preprocessing. The dataset used is the Oxford-IIIT Pet Dataset.

## License
This project is licensed under the MIT License.

