
import cv2
import dlib
import numpy as np
import math
import time
import logging
predictor_path = "shape_predictor_68_face_landmarks (1).dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR = 0.5

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))

POINTS = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS
ALIGN_POINTS = POINTS
OVERLAY_POINTS = [POINTS]

class TimeProfiler(object):
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        logging.info("The %s is done in %fs", self.label, time.time() - self.start)

def get_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return -1
    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])

def get_cam_frame(cam):
    ret, img = cam.read()
    img = cv2.resize(img, (640, 480))
    return img

def glasses_filter(cam, glasses, should_show_bounds=False):
    with TimeProfiler("image capture"):
        face = get_cam_frame(cam)

    with TimeProfiler("face pose prediction"):
        landmarks = get_landmarks(face)

    if type(landmarks) is int:
        return

    # Get the bounding box of the face
    x, y, w, h = cv2.boundingRect(landmarks)

    # Calculate the scaling factor for the glasses
    glasses_scale = w / glasses.shape[1]

    # Resize the glasses to fit the width of the face
    glasses_resized = cv2.resize(glasses, None, fx=glasses_scale, fy=glasses_scale)

    # Calculate the coordinates to place the glasses
    x_offset = x
    y_offset = y

    # Extract the alpha channel from the glasses image
    alpha_channel = glasses_resized[:, :, 3]

    # Create a mask for the glasses
    mask = np.stack([alpha_channel] * 3, axis=2)

    # Extract the RGB channels from the glasses image
    glasses_rgb = glasses_resized[:, :, :3]

    # Overlay the glasses onto the face using the mask
    face_with_glasses = face.copy()
    face_with_glasses[y_offset:y_offset + glasses_resized.shape[0], x_offset:x_offset + glasses_resized.shape[1]] = \
        np.where(mask > 0, glasses_rgb, face_with_glasses[y_offset:y_offset + glasses_resized.shape[0],
                                                          x_offset:x_offset + glasses_resized.shape[1]])

    # Show the result
    cv2.imshow("Glasses Filter", face_with_glasses)

def main():
    cam = cv2.VideoCapture(0)
    glasses = cv2.imread(r"resources\gl.png", cv2.IMREAD_UNCHANGED)

    try:
        while True:
            glasses_filter(cam, glasses)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()


