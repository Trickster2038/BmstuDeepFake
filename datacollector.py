import pandas as pd
import faceparts as fp
import cv2 as cv

def generate_dataframe(filename, shape, frame, frame_scale_percent):
    gray_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    data = {}
    r = pow(
            pow(shape[fp.LEFT_EAR_POINT][0] - shape[fp.RIGHT_EAR_POINT][0], 2) +  
            pow(shape[fp.LEFT_EAR_POINT][1] - shape[fp.RIGHT_EAR_POINT][1], 2), 
            0.5
            )
    rect_size = int((r * frame_scale_percent) / 100.0)
    
    data['filename'] = [filename]
    data['fake'] = [(filename[0:4] == 'fake')]
    data['quality'] = [int(filename.split('_')[-1].split('.')[0])]

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