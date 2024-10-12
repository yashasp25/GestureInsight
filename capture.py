import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Directory to store images for each class
save_dir = "gesture_data"
classes = ['Hello', 'Iloveyou', 'No', 'Yes']  
total_samples = 200  

# Create directories for each class if they don't exist
for class_name in classes:
    os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

# Start capturing images
cap = cv2.VideoCapture(0)
class_id = 0  
num_samples = 0

print(f"Starting to capture gesture images for class: {classes[class_id]}")
print("Press 'space' to capture an image, 'n' to move to the next class, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and capture the region if a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box of the hand
            img_h, img_w, _ = frame.shape
            x_min, y_min = img_w, img_h
            x_max, y_max = 0, 0

            # Loop through all hand landmarks to get the bounding box coordinates
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Expand the bounding box a bit for better context
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_w, x_max + margin)
            y_max = min(img_h, y_max + margin)

            # Extract the hand region from the frame
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Display the cropped hand region on the screen
            cv2.imshow('Hand Region', hand_roi)

            # Capture image when spacebar ('space') is pressed
            if num_samples < total_samples and cv2.waitKey(1) & 0xFF == ord(' '):
                file_name = os.path.join(save_dir, classes[class_id], f"{num_samples}.jpg")
                cv2.imwrite(file_name, hand_roi)
                num_samples += 1
                print(f"Captured image {num_samples} for class {classes[class_id]}")
            elif num_samples >= total_samples:
                print(f"Completed capturing for {classes[class_id]}. Press 'n' for the next class.")

    # Move to the next class when 'n' is pressed
    if cv2.waitKey(1) & 0xFF == ord('n'):
        if num_samples < total_samples:
            print(f"Not enough images for class {classes[class_id]}. {total_samples - num_samples} images remaining.")
        else:
            print(f"Moving to next class.")
            class_id += 1
            num_samples = 0
            if class_id >= len(classes):  # If all classes are completed
                print("All classes have been captured.")
                break
            print(f"Now capturing gesture images for class: {classes[class_id]}")

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
