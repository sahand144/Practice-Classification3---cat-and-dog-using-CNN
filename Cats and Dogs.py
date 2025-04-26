# Importing the libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.api.models import Sequential
from keras.api.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, BatchNormalization
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import random
from PIL import Image

# Setting the path
train_path = r".\train"
val_path = r".\val"

# Function to show random images from the train and val set
def show_random_image():
    dataset_path = input("Enter the dataset path: ").strip()
    category = input("Which category? (cat/dog): ").strip().lower()
    full_path = os.path.join(dataset_path, category)
    
    if not os.path.isdir(full_path):
        print(f"Error: The folder {full_path} does not exist.")
        return
    
    images = os.listdir(full_path)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"No images found in {full_path}.")
        return
    
    random_image = random.choice(images)
    image_path = os.path.join(full_path, random_image)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"{category.capitalize()} - {random_image}")
    plt.axis('off')
    plt.show()

# Call the function to show a random image
show_random_image()

# Setting the data generator with more augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Setting the train and validation generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Compute class weights for imbalanced dataset
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
#printing the class weights
print("Class weights:", class_weights)
print("Class weight dict:", class_weight_dict)


# Setting the model
model = Sequential()

# Adding the layers
model.add(Input(shape=(224, 224, 3)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Second layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Third layer
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Flatten layer
model.add(Flatten())

# Dense layer with L2 regularization
model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compiling the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Applying EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Calculate steps
steps_per_epoch = (train_generator.n + train_generator.batch_size - 1) // train_generator.batch_size  # 275 // 32 = 9
validation_steps = (val_generator.n + val_generator.batch_size - 1) // val_generator.batch_size  # 70 // 32 = 3

# Fitting the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=[early_stopping]
)

# Saving the model
model.save('model.h5')

# Testing
val_generator.reset()
predictions = model.predict(val_generator, steps=len(val_generator))
predictions = (predictions > 0.5).astype(int).flatten()

# Printing the classification report
print(classification_report(val_generator.classes, predictions))

# Printing the confusion matrix
print(confusion_matrix(val_generator.classes, predictions))

# Plotting the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(val_generator.classes, predictions), annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.show()

# Plotting the accuracy and loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.title('Loss')
plt.show()
