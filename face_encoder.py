import face_recognition
import os
import pickle

# Path to the folder containing face images
image_folder = "/pathway/to/images/to/be/encoded"

# Load multiple images and create embeddings
known_encodings = []
for filename in os.listdir(image_folder):
    if filename.endswith(".jpeg"):  # Check for JPEG files
        path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])

# Save the encodings to a .pkl file
with open('face_encodings1.pkl', 'wb') as file:
    pickle.dump(known_encodings, file)

print("Facial encodings have been saved to 'face_encodings1.pkl'")
