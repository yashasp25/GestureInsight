import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import shutil

# Constants
img_height, img_width = 64, 64  
batch_size = 32
original_dataset_dir = r'F:/Dream/Gesture/gesture_data'  
train_dir = r'F:/Dream/Gesture/train'
val_dir = r'F:/Dream/Gesture/val'
total_samples = 200  # Number of samples to capture per class

# Create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to split dataset into training and validation sets
def split_dataset():
    for class_name in os.listdir(original_dataset_dir):
        class_dir = os.path.join(original_dataset_dir, class_name)

        if os.path.isdir(class_dir):  # Check if it's a directory
            # Create directories for each class in train and val
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            # Get all image filenames in the class directory
            images = os.listdir(class_dir)
            random.shuffle(images)

            # Determine the split index
            split_index = int(len(images) * 0.8)  # 80% for training, 20% for validation

            # Move images to training directory
            for img in images[:split_index]:
                shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name))

            # Move images to validation directory
            for img in images[split_index:]:
                shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, class_name))

# Function to augment images
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    return image

# Function to load and preprocess images
def process_image(file_path, label):
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG image
        image = tf.image.resize(image, [img_height, img_width])  # Resize image
        image = image / 255.0  # Normalize to [0, 1]
        
        # Apply augmentation
        image = augment_image(image)

        return image, label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None  # Return None or handle as necessary

# Function to create a dataset from directory
def create_dataset(directory):
    # List of class names and their corresponding labels
    class_names = os.listdir(directory)
    class_labels = {class_name: index for index, class_name in enumerate(class_names)}

    file_paths = []
    labels = []

    # Collect file paths and labels
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for img_file in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, img_file))
            labels.append(class_labels[class_name])

    # Create a TensorFlow dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)  # Parallel processing
    return dataset

# Split dataset into train and validation sets
split_dataset()

# Create datasets
train_dataset = create_dataset(train_dir)
val_dataset = create_dataset(val_dir)

# Shuffle and batch the training dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Batch the validation dataset
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(os.listdir(train_dir)), activation='softmax')  # Adjust output units for number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)


# Save the final model
model.save('gestures_recognition_model.h5')

# Visualization of accuracy and loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
