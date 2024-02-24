# FaceGlassesOverlay
## Real-Time Glasses Filter using OpenCV and dlib

This project implements a real-time glasses filter using OpenCV and dlib in Python. The program captures the webcam feed, detects faces in each frame, and overlays a pair of glasses on detected faces.

## Requirements

- Python 3.x
- OpenCV (cv2)
- dlib
- numpy

## Install the dependencies using pip:
- pip install opencv-python dlib numpy

## Usage

1. Clone the repository:
- git clone https://github.com/HuzaifaManzoor/FaceGlassesOverlay.git
- cd real-time-glasses-filter


2. Download the shape predictor model file (`shape_predictor_68_face_landmarks.dat`) from [dlib's official repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

3. Download a pair of glasses image with an alpha channel (e.g., PNG format) and place it in the `resources` directory.

4. Run the main script:

python run.py


5. Press 'q' to exit the application.

## Customization

- You can adjust the `SCALE_FACTOR`, `FEATHER_AMOUNT`, and `COLOUR_CORRECT_BLUR` parameters in the script to fine-tune the face alignment and overlay process.

- Feel free to replace the default glasses image (`gl.png`) with your own glasses image to customize the filter appearance.

## Acknowledgements

This project was inspired by [Adrian Rosebrock's article](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) on facial landmarks with dlib and OpenCV.



