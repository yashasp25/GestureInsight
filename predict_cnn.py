import cv2
import numpy as np
import tensorflow as tf

img_height, img_width = 64, 64

model = tf.keras.models.load_model('gesture_recognition_model.h5')

class_names = ['Hello', 'Iloveyou', 'No', 'yes']

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame: resize, normalize
    resized_frame = cv2.resize(frame, (img_height, img_width))  # Resize to match the input shape
    normalized_frame = resized_frame / 255.0  # Normalize the frame to [0, 1]
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
    confidence = np.max(predictions)  # Get confidence score

    # Get the corresponding class label
    label = class_names[predicted_class]

    # Display the prediction on the frame
    cv2.putText(frame, f'Gesture: {label} ({confidence*100:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with prediction
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
