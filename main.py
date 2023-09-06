import cv2 as cv
import time

print('===== Init =====')

FILE_PATH = '../ml_fake_videos/fake_freeman_1080.mp4'
SCALE_PERCENT = 50
REAL_TIME_DELAY_24_FPS = 1.0 / 24.0
FPS_LABEL_BORDER_PERCENT = 10 

TEXT_COLOR = (0,255,0)
TEXT_SCALE = 1
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
    
    # resize
    width = int(frame.shape[1] * SCALE_PERCENT / 100.0)
    height = int(frame.shape[0] * SCALE_PERCENT / 100.0)
    dim = (width, height)
    frame_to_show = cv.resize(frame, dim, interpolation = cv.INTER_AREA)

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
    # TODO: check params
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(frame_to_show, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # show
    cv.imshow('Play', frame_to_show)
    time.sleep(REAL_TIME_DELAY_24_FPS)
    if cv.waitKey(1) == ord('q'):
        break

vd.release()
cv.destroyAllWindows()
print("\n===== Final =====")