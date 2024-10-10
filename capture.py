import cv2
import os

# Directory to store images for each class
save_dir = "gestures"
classes = ['Hello', 'Iloveyou', 'No', 'yes']  
total_samples = 200  # Number of samples to capture per class

# Create directories for each class if they don't exist
for class_name in classes:
    os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

# Start capturing images
cap = cv2.VideoCapture(0)
class_id = 0  # Start with the first class
num_samples = 0

print(f"Starting to capture gesture images for class: {classes[class_id]}")
print("Press 'space' to capture an image, 'n' to move to the next class, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame and the current class on the screen
    cv2.putText(frame, f'Class: {classes[class_id]}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Images Captured: {num_samples}/{total_samples}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Frame', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Capture image when spacebar ('space') is pressed
    if key == ord(' '):  # Spacebar to capture image
        if num_samples < total_samples:
            file_name = os.path.join(save_dir, classes[class_id], f"{num_samples}.jpg")
            cv2.imwrite(file_name, frame)
            num_samples += 1
            print(f"Captured image {num_samples} for class {classes[class_id]}")
        else:
            print(f"Completed capturing for {classes[class_id]}. Press 'n' for the next class.")

    # Move to the next class when 'n' is pressed
    if key == ord('n'):
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
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
