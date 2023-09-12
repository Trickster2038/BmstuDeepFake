import pandas as pd
import faceparts as fp
import cv2 as cv
import numpy as np
import cv2 as cv

def generate_dataframe(filename, is_fake, shape, frame, frame_scale_percent):
    gray_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    data = {}
    r = pow(
            pow(shape[fp.LEFT_EAR_POINT][0] - shape[fp.RIGHT_EAR_POINT][0], 2) +  
            pow(shape[fp.LEFT_EAR_POINT][1] - shape[fp.RIGHT_EAR_POINT][1], 2), 
            0.5
            )
    rect_size = int((r * frame_scale_percent) / 100.0)
    
    data['filename'] = [filename]
    data['fake'] = [is_fake]
    # data['quality'] = [int(filename.split('_')[-1].split('.')[0])]

    for zone in fp.FACIAL_LANDMARKS_IDXS.keys():
        for point_id in range(fp.FACIAL_LANDMARKS_IDXS[zone][0], fp.FACIAL_LANDMARKS_IDXS[zone][1]):

            point = shape[point_id]
            point_x = point[0]
            point_y = point[1]
            x1, y1, x2, y2 = \
                int(point_x - int(rect_size/2)), \
                int(point_y - int(rect_size/2)), \
                int(point_x + int(rect_size/2)), \
                int(point_y + int(rect_size/2))
        
            data['pt_' + str(point_id) + '_' + zone +'_raw'] = [gray_frame[y1:y2, x1:x2].ravel()]

    df = pd.DataFrame(data=data)
    return df.copy()

MAX_WIDTH = 720
MAX_HEIGHT = 480
FPS_LABEL_BORDER_PERCENT = 10 
FACE_DETECTION_SIZE_PERCENT = 20

TEXT_COLOR = (0,255,0)
TEXT_SCALE = 3
TEXT_THICKNESS = 2
TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX

HIST_SIZE_PERCENT = 15 # relative to face width

# returns (frame, None) or (frame, df)
def generate_shape(frame, filename, is_fake, frame_n, face_classifier):

    df = None

    frame_to_show = frame.copy()
    frame_to_show_2 = frame.copy()

    gray_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)

    # count frames
    text = str(frame_n)
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
    
    faces = face_classifier.detectMultiScale(gray_frame, 1.1, 5, 
                                             minSize=(
                                                 int(frame_to_show.shape[0] * FACE_DETECTION_SIZE_PERCENT / 100.0), 
                                                 int(frame_to_show.shape[0] * FACE_DETECTION_SIZE_PERCENT / 100.0)))
    
    shapes = []

    for (x, y, w, h) in faces:
        cv.rectangle(frame_to_show, (x, y), (x + w, y + h), TEXT_COLOR, TEXT_SCALE)

        shapes = fp.detect_faces(frame_to_show)

    if len(shapes) > 0:
        df = generate_dataframe(filename, is_fake, shapes[0], frame, HIST_SIZE_PERCENT)
        shp = shapes[0]
        frame_to_show = fp.visualize_facial_landmarks(
            frame_to_show, 
            shp
            )
    
        # scan for hists
        r = pow(
            pow(shp[fp.LEFT_EAR_POINT][0] - shp[fp.RIGHT_EAR_POINT][0], 2) +  
            pow(shp[fp.LEFT_EAR_POINT][1] - shp[fp.RIGHT_EAR_POINT][1], 2), 
            0.5
            )
        rect_size = int((r * HIST_SIZE_PERCENT) / 100.0)

        hists_data = []
        for key in fp.HIST_SCAN_POINTS.keys():
            point = shp[fp.HIST_SCAN_POINTS[key]]
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

        face_parts_frames = []
        for i in range(len(hists_data)):
            data = hists_data[i]
            # print(data)
            x1 = data['area'][0][0]
            y1 = data['area'][0][1]
            x2 = data['area'][1][0]
            y2 = data['area'][1][1]
            face_parts_frames.append(
                cv.cvtColor(gray_frame[y1:y2, x1:x2], cv.COLOR_GRAY2BGR)
                )
            
    k = max(frame_to_show.shape[0] / MAX_HEIGHT, frame_to_show.shape[1] / MAX_WIDTH)
    width = int(frame_to_show.shape[1] / k)
    height = int(frame_to_show.shape[0] / k)
    dim = (width, height)
    frame_to_show = cv.resize(frame_to_show, dim, interpolation = cv.INTER_AREA)
    frame_to_show_2 = cv.resize(frame_to_show_2, dim, interpolation = cv.INTER_AREA)
    multi_frames = np.concatenate((frame_to_show_2, frame_to_show), axis=1)

    if len(shapes)>0:
        face_parts_frames_united = np.concatenate(face_parts_frames, axis=1)
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

    return (multi_frames, df)

# ===============================================

SAVEABLE = True
SAVE_PATH = '../dataset_frames/'
FOLDER_PATH = '../DeepfakeChallenge/train_sample_videos/'
METADATA_PATH = 'metadata.json'
TARGET_FRAME_N = 30
SAMPLE_SIZE = 50
OUTPUT_FILENAME = 'dataframe_total.csv'

if __name__ == "__main__":

    print('===== Init =====')

    df_total = None
    cnt_total = 0

    face_classifier = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    df_fakes = pd.read_json(METADATA_PATH)
    df_fakes = df_fakes.T
    df_fakes = df_fakes.sample(SAMPLE_SIZE)
    df_fakes['filename'] = df_fakes.index
    df_fakes = df_fakes.reset_index()

    cnt_outer = 0

    for index, row in df_fakes.iterrows():
        cnt_outer += 1
        print(f'processing frame #{cnt_outer}')
        is_fake = (row['label'] == 'FAKE')
        filename = row['filename']

        vd = cv.VideoCapture(FOLDER_PATH + filename)
        cnt_inner = 0

        while vd.isOpened() and cnt_inner < TARGET_FRAME_N:

            ret, frame = vd.read()
            cnt_inner += 1

            if cnt_inner == TARGET_FRAME_N:
                frame_result, d = generate_shape(frame, filename, is_fake, cnt_inner, face_classifier)
                if not(d is None):
                    cnt_total += 1
                    print(f'processing frame #{cnt_outer} - ok')
                    if df_total is None:
                        df_total = pd.DataFrame(data=d)
                    else:
                        df_total = pd.concat([df_total, d], ignore_index=True)
                else:
                    print(f'processing frame #{cnt_outer} - no face found')

                if SAVEABLE and not(d is None):
                    cv.imwrite(SAVE_PATH + str(cnt_total) + '_' + filename + '.png', frame_result)
                    
    df_total.to_csv(OUTPUT_FILENAME)
    vd.release()
    cv.destroyAllWindows()
    print(f'\nfaces found: {cnt_total}')
    print("===== Final =====")