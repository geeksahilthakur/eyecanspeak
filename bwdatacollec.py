import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Hand model for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam to capture video
cap = cv2.VideoCapture(0)

# Set parameters for hand image processing
gesture_name = input("Enter the name for the hand gesture (label): ")
folder = f"datasets/{gesture_name}"

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

frame_count = 0
start_time = time.time()
capture_interval = 0.1  # Capture interval between frames (seconds)
imgSize = 300  # The size for the cropped image (300x300)
offset = 20  # Padding around the hand for cropping

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
                x_min, y_min = np.min(lmArray, axis=0) - offset
                x_max, y_max = np.max(lmArray, axis=0) + offset

                # Ensure the bounding box doesn't exceed the frame size
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                # Crop the hand region
                cropped_hand = black_frame[y_min:y_max, x_min:x_max]

                # Resize the cropped hand to 300x300 pixels
                if cropped_hand.size > 0:
                    cropped_hand = cv2.resize(cropped_hand, (imgSize, imgSize))

                # Check if enough time has passed to capture another frame
                current_time = time.time()
                if current_time - start_time >= capture_interval:
                    image_name = f"{gesture_name}_{frame_count}.jpg"
                    cv2.imwrite(os.path.join(folder, image_name), cropped_hand)

                    frame_count += 1
                    start_time = current_time
                    print(f"Frame {frame_count} captured and saved to {folder}/{image_name}")

        else:
            print("No hands detected.")

        # Display the frame count in the OpenCV window
        cv2.putText(black_frame, f"Frames Captured: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the result frame with the hand in white and background in black
        cv2.imshow('Hand Gesture Capture', black_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting the capture loop.")
            break

finally:
    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
