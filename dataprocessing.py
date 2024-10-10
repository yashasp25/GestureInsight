import os
import shutil
import random
import tensorflow as tf

img_height, img_width = 64, 64  
batch_size = 32

#dataset
original_dataset_dir = r'F:/Dream/Gesture/gestures'  

train_dir = r'F:/Dream/Gesture/train'
val_dir = r'F:/Dream/Gesture/val'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

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

# Function to load and preprocess images
def process_image(file_path, label):
    # Load the image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG image
    image = tf.image.resize(image, [img_height, img_width])  # Resize image
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

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

# Create datasets
train_dataset = create_dataset(train_dir)
val_dataset = create_dataset(val_dir)

# Shuffle and batch the training dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Batch the validation dataset
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
