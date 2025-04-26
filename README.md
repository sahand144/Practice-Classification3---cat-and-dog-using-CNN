Cat and Dog Image Classification
Overview
This project implements a binary image classification model to distinguish between cats and dogs using a small dataset (275 training images, 70 validation images). A custom Convolutional Neural Network (CNN) is built to classify the images, addressing challenges like class imbalance and limited data through regularization and data augmentation.
Dataset

Training Set: 275 images (approximately 95 cats, 180 dogs)
Validation Set: 70 images (24 cats, 46 dogs)
Source: Custom dataset stored at D:\datasets\New To Work on 3\Cats and Dogs

Model

Architecture: Custom CNN with 3 convolutional blocks (Conv2D, ReLU, MaxPooling, BatchNormalization, Dropout), followed by Flatten, Dense(256, ReLU, L2 reg), and Dense(1, sigmoid)
Optimizer: Adam (learning rate=0.0001)
Loss: Binary cross-entropy
Regularization: Dropout (0.25), L2 regularization (0.01), data augmentation (shear, zoom, flip, rotation, brightness, shift)
Class Imbalance Handling: Class weights (Cats: 1.45, Dogs: 0.76)

Training

Epochs: Up to 50 with EarlyStopping (patience=5); stopped after 11 epochs
Batch Size: 32
Steps: Training: 9 steps/epoch (275 images); Validation: 3 steps/epoch (70 images)

Results

Accuracy: 63%
F1-score (Cats): 0.19 (precision: 0.38, recall: 0.12)
F1-score (Dogs): 0.76 (precision: 0.66, recall: 0.89)
Confusion Matrix:
True Cats, Predicted Cats: 3
True Cats, Predicted Dogs: 21
True Dogs, Predicted Cats: 5
True Dogs, Predicted Dogs: 41


Observations: The model overfits (training accuracy: 0.66, validation accuracy: 0.34) and performs poorly on Cats due to small dataset size and class imbalance.

How to Run

Ensure dependencies: tensorflow, numpy, matplotlib, scikit-learn, seaborn, pillow.
Update train_path and val_path in the script to your dataset location.
Run the script: python cat_dog_classifier.py.
View random images, training progress, and evaluation metrics (classification report, confusion matrix, accuracy/loss plots).

Future Improvements

Collect more data, especially for Cats, to reduce class imbalance.
Use transfer learning with a pretrained model (e.g., VGG16, ResNet50) to improve feature extraction.
Increase regularization (e.g., higher Dropout, stronger L2 penalty).
Fine-tune hyperparameters (e.g., learning rate, batch size).

