import cv2
import os
import time

class VideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.class_name = ""
        self.capturing = False
        self.counter = 0

    def set_class_name(self, class_name):
        self.class_name = class_name
        self.counter = 0  # Reset counter

    def start_capture(self):
        self.capturing = True

    def stop_capture(self):
        self.capturing = False

    def get_frames(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break

            if self.capturing and self.class_name:
                # Save the frame to the class folder
                img_path = os.path.join(self.class_name, f"Image_{time.time()}.jpg")
                cv2.imwrite(img_path, img)
                self.counter += 1

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def __del__(self):
        self.cap.release()
