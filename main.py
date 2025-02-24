# main.py
import gradio as gr
import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from config import Config
from train import train
from inference import predict, load_and_preprocess_image
from dataset import PetDataset

class PetDetectionUI:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.history = None
        self.load_model()

    def load_model(self):
        if os.path.exists(Config.BEST_MODEL_PATH) and os.path.exists(Config.CLASS_NAMES_PATH):
            try:
                self.model = tf.keras.models.load_model(Config.BEST_MODEL_PATH)
                with open(Config.CLASS_NAMES_PATH, 'r') as f:
                    self.class_names = json.load(f)
                return True
            except:
                return False
        return False

    def save_class_names(self):
        with open(Config.CLASS_NAMES_PATH, 'w') as f:
            json.dump(self.class_names, f)

    def plot_training_history(self, history):
        # Create figure for accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        acc_plot_path = 'graphs/accuracy_plot.png'
        plt.savefig(acc_plot_path)
        plt.close()

        # Create figure for loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_plot_path = 'graphs/loss_plot.png'
        plt.savefig(loss_plot_path)
        plt.close()

        return acc_plot_path, loss_plot_path

    def generate_confusion_matrix(self, dataset):
        # Get predictions for validation set
        y_true = []
        y_pred = []
        
        for images, labels in dataset:
            predictions = self.model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_plot_path = 'graphs/confusion_matrix.png'
        plt.savefig(cm_plot_path)
        plt.close()

        return cm_plot_path

    def calculate_metrics(self, dataset):
        # Get predictions for validation set
        y_true = []
        y_pred = []
        
        for images, labels in dataset:
            predictions = self.model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Create a table of metrics
        metrics_table = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["Precision", f"{precision:.4f}"],
            ["Recall", f"{recall:.4f}"],
            ["F1-Score", f"{f1:.4f}"]
        ]

        return metrics_table

    def detect_pet(self, image):
        if self.model is None:
            return "Please train the model first or load a pre-trained model.", None
        
        try:
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image.save(temp_path)
            
            result = predict(self.model, temp_path, self.class_names)
            os.remove(temp_path)
            
            # Format results
            output = f"Top Prediction: {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.2%})\n\n"
            output += "Top 3 Predictions:\n"
            for pred in result['top_3_predictions']:
                output += f"- {pred['class']}: {pred['confidence']:.2%}\n"
            
            return output, image
        except Exception as e:
            return f"Error during detection: {str(e)}", None

    def train_model(self, 
                   batch_size, 
                   epochs, 
                   learning_rate,
                   dropout_rate,
                   data_augmentation,
                   use_early_stopping,
                   early_stopping_patience,
                   use_lr_reduction,
                   lr_reduction_patience,
                   lr_reduction_factor,
                   validation_split,
                   progress=gr.Progress()):
        try:
            # Update config
            Config.BATCH_SIZE = batch_size
            Config.EPOCHS = epochs
            Config.LEARNING_RATE = learning_rate
            Config.VALIDATION_SPLIT = validation_split
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Define a callback to save the model after each epoch
            class SaveModelCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    model_path = f"models/model_epoch_{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                    self.model.save(model_path)
            
            # Train model
            progress(0, desc="Starting training...")
            self.model, self.history, self.class_names = train(
                dropout_rate=dropout_rate,
                use_augmentation=data_augmentation,
                use_early_stopping=use_early_stopping,
                early_stopping_patience=early_stopping_patience,
                use_lr_reduction=use_lr_reduction,
                lr_reduction_patience=lr_reduction_patience,
                lr_reduction_factor=lr_reduction_factor,
            )
            
            # Save the final model
            final_model_path = Config.BEST_MODEL_PATH
            self.model.save(final_model_path)
            self.save_class_names()
            
            # Generate plots
            acc_plot, loss_plot = self.plot_training_history(self.history)
            
            # Generate confusion matrix
            dataset = PetDataset()
            _, val_ds = dataset.create_datasets()
            cm_plot = self.generate_confusion_matrix(val_ds)
            
            # Calculate metrics
            metrics_table = self.calculate_metrics(val_ds)
            
            # Training summary
            final_accuracy = self.history.history['accuracy'][-1]
            final_val_accuracy = self.history.history['val_accuracy'][-1]
            
            summary = f"""Training completed successfully!
                    Final Training Accuracy: {final_accuracy:.2%}
                    Final Validation Accuracy: {final_val_accuracy:.2%}
                    Model saved as '{final_model_path}'"""
            
            return summary, acc_plot, loss_plot, cm_plot, metrics_table
            
        except Exception as e:
            return f"Error during training: {str(e)}", None, None, None, None

    def create_interface(self):
        with gr.Blocks(title="Breed Detection System") as interface:
            gr.Markdown("# 🐱🐶 Animal Breed Detection System")
            
            with gr.Tabs():
                # Detection Tab
                with gr.Tab("Breed Detection"):
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(type="pil", label="Upload Pet Image")
                            detect_button = gr.Button("Detect Breed")
                        with gr.Column():
                            output_text = gr.Textbox(label="Detection Results")
                            output_image = gr.Image(label="Processed Image")
                    
                    detect_button.click(
                        fn=self.detect_pet,
                        inputs=input_image,
                        outputs=[output_text, output_image]
                    )
                
                # Training Tab
                with gr.Tab("Model Training"):
                    with gr.Row():
                        # Training Parameters
                        with gr.Column():
                            batch_size = gr.Slider(
                                minimum=8,
                                maximum=128,
                                step=8,
                                value=32,
                                label="Batch Size"
                            )
                            epochs = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Number of Epochs"
                            )
                            learning_rate = gr.Slider(
                                minimum=0.0001,
                                maximum=0.01,
                                step=0.0001,
                                value=0.001,
                                label="Learning Rate"
                            )
                            dropout_rate = gr.Slider(
                                minimum=0.0,
                                maximum=0.5,
                                step=0.1,
                                value=0.3,
                                label="Dropout Rate"
                            )
                            validation_split = gr.Slider(
                                minimum=0.1,
                                maximum=0.3,
                                step=0.05,
                                value=0.2,
                                label="Validation Split"
                            )
                            
                        # Advanced Parameters
                        with gr.Column():
                            data_augmentation = gr.Checkbox(
                                label="Use Data Augmentation",
                                value=True
                            )
                            use_early_stopping = gr.Checkbox(
                                label="Use Early Stopping",
                                value=True
                            )
                            early_stopping_patience = gr.Slider(
                                minimum=3,
                                maximum=20,
                                step=1,
                                value=10,
                                label="Early Stopping Patience"
                            )
                            use_lr_reduction = gr.Checkbox(
                                label="Use Learning Rate Reduction",
                                value=True
                            )
                            lr_reduction_patience = gr.Slider(
                                minimum=2,
                                maximum=10,
                                step=1,
                                value=5,
                                label="LR Reduction Patience"
                            )
                            lr_reduction_factor = gr.Slider(
                                minimum=0.1,
                                maximum=0.5,
                                step=0.1,
                                value=0.2,
                                label="LR Reduction Factor"
                            )
                    
                    train_button = gr.Button("Train Model")
                    
                    # Training Results
                    with gr.Row():
                        training_output = gr.Textbox(label="Training Results")
                    
                    with gr.Row():
                        acc_plot = gr.Image(label="Accuracy Plot")
                        loss_plot = gr.Image(label="Loss Plot")
                    
                    with gr.Row():
                        confusion_matrix_plot = gr.Image(label="Confusion Matrix")
                    
                    with gr.Row():
                        metrics_table = gr.Dataframe(
                            headers=["Metric", "Value"],
                            label="Model Metrics"
                        )
                    
                    train_button.click(
                        fn=self.train_model,
                        inputs=[
                            batch_size,
                            epochs,
                            learning_rate,
                            dropout_rate,
                            data_augmentation,
                            use_early_stopping,
                            early_stopping_patience,
                            use_lr_reduction,
                            lr_reduction_patience,
                            lr_reduction_factor,
                            validation_split
                        ],
                        outputs=[
                            training_output,
                            acc_plot,
                            loss_plot,
                            confusion_matrix_plot,
                            metrics_table
                        ]
                    )
            
            gr.Markdown("""
                ## Instructions
                1. **Pet Breed Detection Tab**: Upload an image and click 'Detect Breed' to identify the pet breed
                2. **Model Training Tab**: Adjust training parameters and click 'Train Model' to start training
                
                The training tab now includes advanced parameters and visualizations:
                - Accuracy and loss plots show training progress
                - Confusion matrix shows model performance across classes
                - Metrics table displays F1-score, Precision, Recall, and Accuracy
                - Advanced parameters allow fine-tuning of the training process
            """)
        
        return interface

def main():
    ui = PetDetectionUI()
    interface = ui.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
