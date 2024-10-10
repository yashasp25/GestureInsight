import tensorflow as tf
from tensorflow.keras import layers, models

img_height, img_width = 64, 64  
batch_size = 32

# Load the datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'F:/Dream/Gesture/train',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'F:/Dream/Gesture/val',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  
)

model.save('gesture_recognition_model.h5')
