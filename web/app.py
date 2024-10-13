import zipfile
from cvzone.HandTrackingModule import HandDetector
from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import os
import numpy as np
import math

app = Flask(__name__)

# Global variables
capture = False
class_name = ''
counter = 0
output_dir = 'static/images'
offset = 20
imgSize = 300

# Create folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to handle camera feed
def generate_frames():
    global capture, class_name, counter
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            continue  # Skip to the next iteration if frame capture fails

        hands, img = detector.findHands(img)

        if hands and capture:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Check if imgCrop is not empty before resizing
            if imgCrop.size > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Save the image with the current counter if capturing
                counter += 1
                cv2.imwrite(os.path.join(output_dir, class_name, f'Image_{counter}.jpg'), imgWhite)
                print(f"Image saved: {counter}")

            # Yield the image for displaying in the browser
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            # Yield the frame without modifications if no hands are detected or not capturing
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html', class_name=class_name, counter=counter)


# Route for video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_class_name', methods=['POST'])
def set_class_name():
    global class_name, counter
    data = request.json
    class_name = data['class_name']
    counter = 0  # Reset counter when class name is changed
    if not os.path.exists(os.path.join(output_dir, class_name)):
        os.makedirs(os.path.join(output_dir, class_name))  # Create directory for class if it doesn't exist
    return jsonify({'status': 'success', 'class_name': class_name})

# Route to get the image count for the current class
@app.route('/get_image_count')
def get_image_count():
    global class_name, counter
    images = []

    # Get the list of images in the specified class directory
    if class_name:
        class_dir = os.path.join(output_dir, class_name)
        images = [os.path.join('static/images', class_name, f) for f in os.listdir(class_dir) if f.endswith('.jpg')]
        counter = len(images)

    return jsonify({'counter': counter, 'images': images})


# Route to start capturing
@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture
    capture = True
    return jsonify({'status': 'Capturing started'})


# Route to stop capturing
@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capture
    capture = False
    return jsonify({'status': 'Capturing stopped'})


# Route to download images as a zip file
@app.route('/download_images')
def download_images():
    global class_name

    # Create a zip file containing images
    zip_filename = f"{class_name}_images.zip"
    zip_path = os.path.join(output_dir, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(os.path.join(output_dir, class_name)):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    return send_file(zip_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
