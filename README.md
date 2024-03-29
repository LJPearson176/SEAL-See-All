# Project SEAL-See All
Project SEAL-See All is a comprehensive computer vision application designed to augment the interaction between users and computers using facial detection, hand gesture recognition, and object detection. Implemented on a macOS environment and leveraging a webcam, SEAL enables users to control actions or trigger scripts through specific hand gestures while also offering object detection and text extraction capabilities.

Features
Facial Detection and Recognition: Identifies and recognizes faces in real-time video streams.
Hand Gesture Recognition: Detects hand gestures including closed fist, open palm, pointing up, thumbs up, thumbs down, and victory sign using MediaPipe.
Object Detection: Utilizes YOLO COCO dataset for recognizing a wide range of common objects.
Optical Character Recognition (OCR): Extracts text from identified objects with high confidence using PyTesseract.

Installation
Ensure you have Python 3.x installed on your macOS. Clone the repository and navigate to the project directory.

git clone <repository-url>
cd SEAL-See-All


Dependencies

Install the required Python packages:
pip install -r requirements.txt


Make sure to have the models and required files in the following paths:
YOLO configuration and weights: /yolo/yolov3.cfg, /yolo/yolov3.weights
COCO names: /yolo/coco.names
Haar Cascade for face detection: /haar/haarcascade_frontalface_default.xml
Gesture recognizer model: gesture_recognizer/gesture_recognizer.task
Tesseract OCR: Ensure tesseract is installed and correctly set up on your macOS.


Preparing Facial Encodings
Place sample images in the designated folder.
Run image_augmentation.py to augment your images for better recognition performance.
Use face_encoder.py to generate facial encoding files.

python image_augmentation.py
python face_encoder.py

![image](https://github.com/LJPearson176/SEAL-See-All/assets/145518111/57558ccc-ecb7-4826-9536-edac0eb6da44)



Usage
Run the main script to start the application:

python seal.py

Toggle Object Detection: Press the 'o' key to turn on/off object detection.
Extract Text: Press the 'a' key to activate OCR on objects detected with over 80% confidence.

Yolo Objects:

names:

  0: person
  
  1: bicycle
  
  2: car
  
  3: motorcycle
  
  4: airplane
  
  5: bus
  
  6: train
  
  7: truck
  
  8: boat
  
  9: traffic light
  
  10: fire hydrant
  
  11: stop sign
  
  12: parking meter
  
  13: bench
  
  14: bird
  
  15: cat
  
  16: dog
  
  17: horse
  
  18: sheep
  
  19: cow
  
  20: elephant
  
  21: bear
  
  22: zebra
  
  23: giraffe
  
  24: backpack
  
  25: umbrella
  
  26: handbag
  
  27: tie
  
  28: suitcase
  
  29: frisbee
  
  30: skis
  
  31: snowboard
  
  32: sports ball
  
  33: kite
  
  34: baseball bat
  
  35: baseball glove
  
  36: skateboard
  
  37: surfboard
  
  38: tennis racket
  
  39: bottle
  
  40: wine glass
  
  41: cup
  
  42: fork
  
  43: knife
  
  44: spoon
  
  45: bowl
  
  46: banana
  
  47: apple
  
  48: sandwich
  
  49: orange
  
  50: broccoli
  
  51: carrot
  
  52: hot dog
  
  53: pizza
  
  54: donut
  
  55: cake
  
  56: chair
  
  57: couch
  
  58: potted plant
  
  59: bed
  
  60: dining table
  
  61: toilet
  
  62: tv
  
  63: laptop
  
  64: mouse
  
  65: remote
  
  66: keyboard
  
  67: cell phone
  
  68: microwave
  
  69: oven
  
  70: toaster
  
  71: sink
  
  72: refrigerator
  
  73: book
  
  74: clock
  
  75: vase
  
  76: scissors
  
  77: teddy bear
  
  78: hair drier
  
  79: toothbrush


![image](https://github.com/LJPearson176/SEAL-See-All/assets/145518111/a7701152-303d-4d70-b282-5b3967f8a4a0)


![image](https://github.com/LJPearson176/SEAL-See-All/assets/145518111/243ff491-e24a-4006-8e09-5a2ccb99086e)

![image](https://github.com/LJPearson176/SEAL-See-All/assets/145518111/dfff73a5-3aab-43dc-9d21-3e7ac22bc823)

![image](https://github.com/LJPearson176/SEAL-See-All/assets/145518111/6a70b840-e58c-4a50-9495-39190bc6b5c7)

![image](https://github.com/LJPearson176/SEAL-See-All/assets/145518111/920f465d-75ce-4f82-afba-fc5d167a0f70)



Adding Gesture Actions

Edit the gesture_recognition_callback function in seal.py to define actions for detected gestures. 


To-Do:

Image Enhancement Integration: 

I plan to integrate an image enhancement model to refine the images captured for object detection. This enhancement aims to significantly improve the quality of images before they are processed by OCR, thereby enhancing the accuracy of text extraction from complex backgrounds or low-quality images.


Voice Command Activation: 

In an effort to broaden the interactivity of SEAL-See All, I am working on incorporating a feature that enables live voice recording with immediate transcription. The transcribed audio will then be fed into ShellGPT, facilitating the conversion of voice commands into actionable code execution within the command line environment. This enhancement is targeted towards creating a more seamless and hands-free user experience, enabling users to execute commands and control the application through voice interactions.

Stay tuned for these exciting updates as we continue to enhance the functionality and user experience of Project SEAL-See All.
