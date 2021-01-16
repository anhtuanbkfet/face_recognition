# face_recognition
Human face recognition using opencv, dlib, and keras

# How to use:
1. Install all package as requirements.txt file defined
2. DATASET FOLDER TREE: 
DATASET
    + NAMED
        + USER 1
        + USER 2
        + ...

    + UNKNOWN

Dataset folder is used to save face images, to collect face images, go 'face_detection.py' file and change save_faces = True

To training face recognize model, run:
    python face_recognition.py

To run test app, run:
    python face_detection.py (remember that set save_faces = False firstly)

    All faces that detect by unknown label, will be saved to DATASET/UNKNOWN folder while app is running.
