import face_recognition
import time
import cv2
import os
import pickle
import numpy as np
import csv
import pytesseract
import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess
import time

# Global variable to track the start time of a gesture
gesture_start_time = {}
# Threshold time to trigger action (in seconds)
trigger_time_threshold = 2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directory for saving OCR images
ocr_images_dir = '/Users/ljp176/brio/ocr_images'
os.makedirs(ocr_images_dir, exist_ok=True)

# Function to perform OCR
def perform_ocr(frame, box):
    x, y, w, h = box
    x = max(0, x)
    y = max(0, y)
    w = min(frame.shape[1] - x, w)
    h = min(frame.shape[0] - y, h)
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    inverted = cv2.bitwise_not(eroded)
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    tesseract_config = '-l eng --oem 1 --psm 6'
    text = pytesseract.image_to_string(inverted, config=tesseract_config)
    return text

# Load YOLO model
net = cv2.dnn.readNetFromDarknet('/Users/ljp176/Desktop/yolo-coco/yolov3.cfg', 
                                 '/Users/ljp176/Desktop/yolo-coco/yolov3.weights')
layer_names = net.getLayerNames()
output_layer_ids = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layer_ids.flatten()]

# Load COCO class names
with open('/Users/ljp176/Desktop/yolo-coco/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('/Users/ljp176/anaconda3/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Try loading known face encodings
try:
    with open('/Users/ljp176/brio/face_encodings1.pkl', 'rb') as file:
        known_encodings = pickle.load(file)
except FileNotFoundError:
    known_encodings = []

# Function to convert OpenCV frame to MediaPipe Image
def convert_to_mp_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb_frame))

def process_frame_for_gesture(frame, recognizer):
    mp_image = convert_to_mp_image(frame)
    current_time_ms = int(time.time() * 1000)
    recognizer.recognize_async(mp_image, current_time_ms)
 

# Gesture callback functions
def handle_closed_fist():
    print("Gesture Recognized: Closed Fist")

def handle_open_palm():
    print("Gesture Recognized: Open Palm")

def handle_pointing_up():
    print("Gesture Recognized: Pointing Up")

def handle_thumb_up():
    print("Gesture Recognized: Thumb Up")

def handle_thumb_down():
    print("Gesture Recognized: Thumb Down")

def execute_script(script_path):
    try:
        subprocess.run(['python3', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")    

script_path = '/Users/ljp176/Desktop/brio/signout.py'

def handle_victory():
    print("Victory gesture recognized. Executing script...")
    execute_script(script_path)

def process_frame_for_gesture(frame, recognizer):
    mp_image = convert_to_mp_image(frame)
    current_time_ms = int(time.time() * 1000)
    recognizer.recognize_async(mp_image, current_time_ms)


# Define the callback function here
def gesture_recognition_callback(result, frame, timestamp_ms):
    global gesture_start_time
    current_time = time.time()
    gesture_detected = False  # Added flag to check if a gesture is detected

    if result and hasattr(result, 'gestures'):
        for gesture_result in result.gestures:
            if isinstance(gesture_result, list) and gesture_result:
                first_category = gesture_result[0]
                if hasattr(first_category, 'category_name'):
                    gesture_name = first_category.category_name
                    gesture_detected = True  # Set flag to True

                    # Start or update the timer for the detected gesture
                    if gesture_name not in gesture_start_time:
                        gesture_start_time[gesture_name] = current_time

                    # Check if the gesture has been held for the trigger time threshold
                    if current_time - gesture_start_time[gesture_name] >= trigger_time_threshold:
                        handle_gesture(gesture_name)
                        gesture_start_time.pop(gesture_name, None)
                else:
                    print("Category object does not have a category_name attribute")
            else:
                print("Gesture result is either empty or not a list")
    else:
        print("No gesture results or result object lacks 'gestures' attribute")

    # Reset timers for gestures that are not currently detected
    if not gesture_detected:  # Check if no gesture was detected in this frame
        gesture_start_time.clear()  # Reset all gesture timers

def handle_gesture(gesture_name):
    # Implement actions for each gesture
    if gesture_name == 'Closed_Fist':
        handle_closed_fist()
    elif gesture_name == 'Open_Palm':
        handle_open_palm()
    elif gesture_name == 'Pointing_Up':
        handle_pointing_up()
    elif gesture_name == 'Thumb_Up':
        handle_thumb_up()
    elif gesture_name == 'Thumb_Down':
        handle_thumb_down()
    elif gesture_name == 'Victory':
        handle_victory()         



# Load gesture recognizer model and set up options
model_path = '/Users/ljp176/Downloads/gesture_recognizer.task'
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=gesture_recognition_callback
)
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)


# Webcam capture
cap = cv2.VideoCapture(0)  # This line initializes the webcam

# Initialize YOLO enabled state
yolo_enabled = False


# Open a CSV file to write OCR results
csv_file_path = '/Users/ljp176/brio/ocr_results1.csv'
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'Object Type', 'Confidence Score', 'OCR Text', 'Box Coordinates', 'Image Path'])



# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video")
        break

    # Initialize face_recognized as False for each frame
    face_recognized = False

    # Toggle YOLO Object Detection with 'o' key
    if cv2.waitKey(1) & 0xFF == ord('o'):
        yolo_enabled = not yolo_enabled

    # Initialize lists for YOLO detection
    boxes = []
    confidences = []
    class_ids = []

    # YOLO Object Detection (if enabled)
    if yolo_enabled:
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = [0, 255, 0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.2f}'.format(label, confidence)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Haar Cascade Face Detection and Recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        current_face_encodings = face_recognition.face_encodings(rgb_face)
        if current_face_encodings:
            matches = face_recognition.compare_faces(known_encodings, current_face_encodings[0])
            if True in matches:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Lucas', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    
                face_recognized = True

    # MediaPipe Hands Detection if face is recognized
    if face_recognized:
        # Process frame for gesture recognition
        process_frame_for_gesture(frame, recognizer)

    # OCR functionality (only if YOLO is enabled and confidence is over 80%)
    if yolo_enabled and cv2.waitKey(1) & 0xFF == ord('a'):
        for i, box in enumerate(boxes):
            if confidences[i] > 0.8:
                ocr_text = perform_ocr(frame, box)
                if ocr_text.strip():
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    object_type = classes[class_ids[i]]
                    confidence_score = confidences[i]

                    # Extract original coordinates for image saving
                    x, y, w, h = box
                    x = max(0, x)
                    y = max(0, y)


                    # Save the image
                    img_name = f'ocr_image_{timestamp.replace(":", "-")}_{i}.jpg'
                    img_path = os.path.join(ocr_images_dir, img_name)
                    cv2.imwrite(img_path, frame[y:y+h, x:x+w])  


                    csv_writer.writerow([timestamp, object_type, confidence_score, ocr_text, box, img_path])
                    print("OCR Result:", ocr_text)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
csv_file.close()