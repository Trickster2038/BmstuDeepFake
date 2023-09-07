import cv2 as cv
import time
import numpy as np
import faceparts

print('===== Init =====')

FILE_PATH = '../ml_fake_videos/fake_freeman_1080.mp4'
SCALE_PERCENT = 30
REAL_TIME_DELAY_24_FPS = 1.0 / 24.0
FPS_LABEL_BORDER_PERCENT = 10 

FACE_DETECTION_SIZE_PERCENT = 20
WAIT_KEY = 1 # 0 - play pressing any key, 1 - autoplay

TEXT_COLOR = (0,255,0)
TEXT_SCALE = 3
TEXT_THICKNESS = 2
TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX


face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

vd = cv.VideoCapture(FILE_PATH)
frame_cnt = 0

# ===== MAIN =====
while vd.isOpened():
    ret, frame = vd.read()
    frame_cnt += 1
    
    frame_to_show = frame

    # count frames
    text = str(frame_cnt)
    coordinates = \
        (
            int(frame_to_show.shape[0] * FPS_LABEL_BORDER_PERCENT / 100.0), # width
            int(frame_to_show.shape[0] * FPS_LABEL_BORDER_PERCENT / 100.0) # height
        )
    frame_to_show = cv.putText(
        frame_to_show, 
        text, 
        coordinates, 
        TEXT_FONT, 
        TEXT_SCALE, 
        TEXT_COLOR, 
        TEXT_THICKNESS, 
        cv.LINE_AA)

    # face recognition

    gray_image = cv.cvtColor(frame_to_show, cv.COLOR_BGR2GRAY)
    # TODO: check params (1.1, 5 - default)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, 
                                             minSize=(
                                                 int(frame_to_show.shape[0] * FACE_DETECTION_SIZE_PERCENT / 100.0), 
                                                 int(frame_to_show.shape[0] * FACE_DETECTION_SIZE_PERCENT / 100.0)))
    for (x, y, w, h) in faces:
        cv.rectangle(frame_to_show, (x, y), (x + w, y + h), TEXT_COLOR, TEXT_SCALE)
    gray_image = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)

    # detect faceparts
    shapes = faceparts.detect_faces(frame_to_show)
    frame_to_show = faceparts.visualize_facial_landmarks(
        frame_to_show, 
        shapes[0]
        )

    # resize
    width = int(frame_to_show.shape[1] * SCALE_PERCENT / 100.0)
    height = int(frame_to_show.shape[0] * SCALE_PERCENT / 100.0)
    dim = (width, height)
    frame_to_show = cv.resize(frame_to_show, dim, interpolation = cv.INTER_AREA)
    gray_image = cv.resize(gray_image, dim, interpolation = cv.INTER_AREA)

    # show
    multi_frames = np.concatenate((frame_to_show, gray_image), axis=1)
    cv.imshow('Play', multi_frames)
    if WAIT_KEY == 1:
        time.sleep(REAL_TIME_DELAY_24_FPS * 4)
    if cv.waitKey(WAIT_KEY) == ord('q'):
        break

vd.release()
cv.destroyAllWindows()
print("\n===== Final =====")