import cv2 as cv
import time
import numpy as np
import faceparts
from matplotlib import pyplot as plt

print('===== Init =====')

plt.ion()

FILE_PATH = '../ml_fake_videos/fake_freeman_1080.mp4'
# SCALE_PERCENT = 30
MAX_WIDTH = 720
MAX_HEIGHT = 480
REAL_TIME_DELAY_24_FPS = 1.0 / 24.0
DELAY = REAL_TIME_DELAY_24_FPS * 24 * 10
FPS_LABEL_BORDER_PERCENT = 10 

FACE_DETECTION_SIZE_PERCENT = 20
WAIT_KEY = 0 # 0 - play pressing any key, 1 - autoplay

TEXT_COLOR = (0,255,0)
TEXT_SCALE = 3
TEXT_THICKNESS = 2
TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX

HIST_SIZE_PERCENT = 100 # relative to face width

global fig, axs
fig, axs = plt.subplots(len(faceparts.HIST_SCAN_POINTS), 1)

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

vd = cv.VideoCapture(FILE_PATH)
frame_cnt = 0

# ===== MAIN =====
while vd.isOpened():
    print(f'~ prepare frame #{frame_cnt}')
    ret, frame = vd.read()
    frame_cnt += 1
    
    frame_to_show = frame.copy()
    frame_to_show_2 = frame.copy()

    gray_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)

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

    print(f'~ face recognition type 1 #{frame_cnt}')
    # face recognition
    # TODO: check params (1.1, 5 - default)
    faces = face_classifier.detectMultiScale(gray_frame, 1.1, 5, 
                                             minSize=(
                                                 int(frame_to_show.shape[0] * FACE_DETECTION_SIZE_PERCENT / 100.0), 
                                                 int(frame_to_show.shape[0] * FACE_DETECTION_SIZE_PERCENT / 100.0)))
    for (x, y, w, h) in faces:
        cv.rectangle(frame_to_show, (x, y), (x + w, y + h), TEXT_COLOR, TEXT_SCALE)
    # TODO: NOTICE gray -> colored
    # gray_frame_colored = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)

    print(f'~ face recognition type 2 #{frame_cnt}')
    # detect faceparts
    shapes = faceparts.detect_faces(frame_to_show)
    if len(shapes) > 0:
        shp = shapes[0]
        frame_to_show = faceparts.visualize_facial_landmarks(
            frame_to_show, 
            shp
            )
    
        # scan for hists
        r = pow(
            pow(shp[faceparts.LEFT_EAR_POINT][0] - shp[faceparts.RIGHT_EAR_POINT][0], 2) +  
            pow(shp[faceparts.LEFT_EAR_POINT][1] - shp[faceparts.RIGHT_EAR_POINT][1], 2), 
            0.5
            )
        rect_size = int((r * HIST_SIZE_PERCENT) / 100.0)

        hists_data = []
        for key in faceparts.HIST_SCAN_POINTS.keys():
            point = shp[faceparts.HIST_SCAN_POINTS[key]]
            point_x = point[0]
            point_y = point[1]
            x1, y1, x2, y2 = \
                int(point_x - int(rect_size/2)), \
                int(point_y - int(rect_size/2)), \
                int(point_x + int(rect_size/2)), \
                int(point_y + int(rect_size/2))
                
            # print(">>",int(rect_size/2),x1,x2,y1,y2,(x1-x2),(y1-y2))
            # FIXME: zero-safeness for coords
            cv.rectangle(
                frame_to_show_2, 
                (x1, y1), 
                (x2, y2), 
                TEXT_COLOR, 
                TEXT_SCALE
                )
            
            # TODO: normalized hist or not?

            # counts, bins = np.histogram(x)
            # plt.stairs(counts, bins)
            hists_data.append({'area': [(x1, y1), (x2, y2)], 'label': key})

        print(f'~ hists for face recognition type 2 #{frame_cnt}')
        face_parts_frames = []
        for i in range(len(hists_data)):
            data = hists_data[i]
            # print(data)
            x1 = data['area'][0][0]
            y1 = data['area'][0][1]
            x2 = data['area'][1][0]
            y2 = data['area'][1][1]
            axs[i].set_title(data['label'] + '_' + str(frame_cnt))
            axs[i].hist(gray_frame[y1:y2, x1:x2].ravel(),256,[0,256])
            face_parts_frames.append(
                cv.cvtColor(gray_frame[y1:y2, x1:x2], cv.COLOR_GRAY2BGR)
                )
        fig.canvas.draw()
        fig.canvas.flush_events()   

    print(f'~ render #{frame_cnt}')

    # resize
    k = max(frame_to_show.shape[0] / MAX_HEIGHT, frame_to_show.shape[1] / MAX_WIDTH)
    width = int(frame_to_show.shape[1] / k)
    height = int(frame_to_show.shape[0] / k)
    dim = (width, height)
    frame_to_show = cv.resize(frame_to_show, dim, interpolation = cv.INTER_AREA)
    frame_to_show_2 = cv.resize(frame_to_show_2, dim, interpolation = cv.INTER_AREA)

    # show
    face_parts_frames_united = np.concatenate(face_parts_frames, axis=1)
    multi_frames = np.concatenate((frame_to_show, frame_to_show_2), axis=1)
    face_parts_frames_united = cv.resize(
        face_parts_frames_united, 
        (
            #       x1 / y1 = x2 / y2
            # =>    y2 = x2 * y1 / x1
            multi_frames.shape[1],
            int(float(multi_frames.shape[1]) * face_parts_frames_united.shape[0] / face_parts_frames_united.shape[1]),
        ), 
        interpolation = cv.INTER_AREA)
    multi_frames = np.concatenate((multi_frames, face_parts_frames_united), axis=0)
    cv.imshow('Play', multi_frames)

    print(f'~ waiting key #{frame_cnt}')
    if WAIT_KEY == 1:
        time.sleep(DELAY)
    if cv.waitKey(WAIT_KEY) == ord('q'):
        plt.pause(0.1) 
        break 

vd.release()
cv.destroyAllWindows()
print("\n===== Final =====")