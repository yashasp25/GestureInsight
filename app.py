import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('gestures_recognition_model.h5')

# Define the class labels
class_labels = {
    0: "Hello",
    1: "Iloveyou",
    2: "No",
    3: "Yes"
}

# Preprocessing function
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Resize to target size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

# Prediction via file upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Load and preprocess the image
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image, target_size=(64, 64))  # Resize to (64, 64)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    
    # Return class name instead of index
    predicted_class_name = class_labels.get(predicted_class, "Unknown")
    
    return jsonify({'predicted_class': predicted_class_name})

# Live prediction via webcam
@app.route('/live-predict', methods=['POST'])
def live_predict():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data"}), 400
    
    # Image is sent in base64 format, decode it
    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocess and predict
    processed_image = preprocess_image(image, target_size=(64, 64))  # Resize to (64, 64)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    
    # Return class name instead of index
    predicted_class_name = class_labels.get(predicted_class, "Unknown")
    
    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
