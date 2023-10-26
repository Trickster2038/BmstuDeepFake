import cv2 as cv
import time
import numpy as np
import faceparts
# from matplotlib import pyplot as plt
import datacollector
import pandas as pd

print('===== Init =====')

# plt.ion()

# FILE_PATH = '../ml_fake_videos/fake_freeman_1080.mp4'
# FILE_PATH = '../ml_fake_videos/real_icc_court_1080.mp4'
# FILE_PATH = '../ml_fake_videos/real_poperechnii_1080.mp4'
# FILE_PATH = '../ml_fake_videos/fake_obama_720.mp4'
FILE_PATH = '../deep_fake_src/ml_fake_videos/fake_backtofuture_1080.mp4'
# FILE_PATH = '../ml_fake_videos/fake_gin_1080.mp4'
# FILE_PATH = '../deep_fake_src/ml_fake_videos/fake_joker_1080.mp4'
# FILE_PATH = '../deep_fake_src/ml_fake_videos/fake_terminator_720.mp4'
# FILE_PATH = '../deep_fake_src/ml_fake_videos/real_poperechnii_1080.mp4'
# FILE_PATH = '../deep_fake_src/ml_fake_videos/fake_freeman_1080.mp4'

# FILE_PATH = '../deep_fake_src/dfdc_train_part_27/aecmpgzdbs.mp4' # fake

# SCALE_PERCENT = 30
MAX_WIDTH = 480
MAX_HEIGHT = 240
REAL_TIME_DELAY_24_FPS = 1.0 / 24.0
DELAY = REAL_TIME_DELAY_24_FPS * 24 * 10
FPS_LABEL_BORDER_PERCENT = 10 

FACE_DETECTION_SIZE_PERCENT = 20
WAIT_KEY = 0 # 0 - play pressing any key, 1 - autoplay

TEXT_COLOR = (0,255,0)
TEXT_SCALE = 3
TEXT_THICKNESS = 2
TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX

HIST_SIZE_PERCENT = 15 # relative to face width

# global fig, axs
# fig, axs = plt.subplots(len(faceparts.HIST_SCAN_POINTS), 1)
# fig.set_size_inches(len(faceparts.HIST_SCAN_POINTS), 7)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

NOISE_REGION_SIZE = 3
NOISE_SIGMA = 60

vd = cv.VideoCapture(FILE_PATH)
frame_cnt = 0

first_frame = False
df = None

# ===== MAIN =====
while vd.isOpened():
    print(f'~ prepare frame #{frame_cnt}')
    ret, frame = vd.read()
    frame_cnt += 1
    
    frame_to_show = frame.copy()
    frame_to_show_2 = frame.copy()

    gray_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    img_filtered_1 = cv.medianBlur(frame_to_show_2, NOISE_REGION_SIZE)
    img_filtered_2 = cv.GaussianBlur(frame_to_show_2, (NOISE_REGION_SIZE, NOISE_REGION_SIZE), 0)
    img_filtered_3 = cv.bilateralFilter(frame_to_show_2, NOISE_REGION_SIZE, NOISE_SIGMA, NOISE_SIGMA)

    # count frames
    text = str(frame_cnt)

    print(f'~ face recognition type 1 #{frame_cnt}')

    print(f'~ face recognition type 2 #{frame_cnt}')

    print(f'~ render #{frame_cnt}')

    # resize
    k = max(frame_to_show.shape[0] / MAX_HEIGHT, frame_to_show.shape[1] / MAX_WIDTH)
    width = int(frame_to_show.shape[1] / k)
    height = int(frame_to_show.shape[0] / k)
    dim = (width, height)
    frame_to_show = cv.resize(frame_to_show, dim, interpolation = cv.INTER_AREA)
    frame_to_show_2 = cv.resize(frame_to_show_2, dim, interpolation = cv.INTER_AREA)
    img_filtered_1 = cv.resize(img_filtered_1, dim, interpolation = cv.INTER_AREA)
    img_filtered_2 = cv.resize(img_filtered_2, dim, interpolation = cv.INTER_AREA)
    img_filtered_3 = cv.resize(img_filtered_3, dim, interpolation = cv.INTER_AREA)
    # show
    multi_frames_1 = np.concatenate((img_filtered_1 - frame_to_show, frame_to_show), axis=1)
    multi_frames_2 = np.concatenate((img_filtered_2 - frame_to_show, img_filtered_3 - frame_to_show), axis=1)
    multi_frames = np.concatenate((multi_frames_1, multi_frames_2), axis=0)
    cv.imshow('Play', multi_frames)

    if not first_frame:
        print(f'~ waiting key #{frame_cnt}')
        if WAIT_KEY == 1:
            time.sleep(DELAY)
        if cv.waitKey(WAIT_KEY) == ord('q'):
            break 
        # plt.close('all')

    first_time = False

vd.release()
cv.destroyAllWindows()
print("\n===== Final =====")