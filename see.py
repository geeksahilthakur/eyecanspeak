import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf  # Import TensorFlow

# Load the pre-trained model
model_path = 'bwmodel/keras_model.h5'  # Update this with the path to your model
model = tf.keras.models.load_model(model_path)

# Load label mapping
label_path = 'bwmodel/labels.txt'  # Update this with the path to your label file
with open(label_path, 'r') as file:
    labels = file.read().splitlines()  # Read labels from the file

# Initialize MediaPipe Hand model for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam to capture video
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture each frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Get the frame height and width
        h, w, _ = frame.shape

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand detection
        result = hands.process(rgb_frame)

        # Create a completely black background
        black_frame = np.zeros_like(frame)

        # If hands are detected, process the landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the hand landmarks in white on the black background
                mp_drawing.draw_landmarks(
                    black_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # White color for landmarks
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)  # White color for hand connections
                )

                # Get bounding box for the hand landmarks
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    # Convert normalized landmarks to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([cx, cy])

                # Get the min and max coordinates for the hand bounding box
                lmArray = np.array(lmList)
                x_min, y_min = np.min(lmArray, axis=0)
                x_max, y_max = np.max(lmArray, axis=0)

                # Crop the hand region
                cropped_hand = frame[y_min:y_max, x_min:x_max]

                # Resize the cropped hand to 224x224 pixels (to match model input shape)
                if cropped_hand.size > 0:
                    cropped_hand = cv2.resize(cropped_hand, (224, 224))

                    # Normalize the image for the model
                    cropped_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB) / 255.0
                    cropped_hand = np.expand_dims(cropped_hand, axis=0)  # Add batch dimension

                    # Make predictions
                    predictions = model.predict(cropped_hand)
                    predicted_class = np.argmax(predictions)  # Get the index of the class with the highest probability
                    predicted_label = labels[predicted_class]  # Get the corresponding label

                    # Display the prediction on the frame
                    cv2.putText(black_frame, f"Predicted: {predicted_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:
            print("No hands detected.")

        # Show the result frame with the hand in white and background in black
        cv2.imshow('Hand Gesture Detection', black_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting the detection loop.")
            break

finally:
    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
