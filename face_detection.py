import cv2
import dlib
import time
import glob
import pandas as pd
import numpy as np
from scipy.spatial import distance

from face_alignment import AlignDlib
from model import create_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#######################################
# setup
save_faces = False
recognize_face = True
save_output_video = False
threshold = 0.65
scale_factor = 4
out_save_dir = 'DATASET/UNKNOWN'
#######################################
# define face detector:
face_cascade = cv2.CascadeClassifier('pre_models/haarcascade_frontalface_default.xml')

def detect_faces(img, scale=1):
    time_start = time.time()
    faces = face_cascade.detectMultiScale(img, 1.23, 5)
    faces = [dlib.rectangle(scale*face[0], scale*face[1], scale*(face[0]+face[2]), scale*(face[1]+face[3]))  for face in faces]
    time_elapsed = time.time()-time_start
    return faces, time_elapsed

def align_face(alignment, face):
    #print(img.shape)
    (h,w,c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    #print(bb)
    return alignment.align(96, face, bb,landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def load_data_embs():
    train_paths = glob.glob("DATASET/NAMED/*")
    nb_classes = len(train_paths)

    df_train = pd.DataFrame(columns=['image', 'label', 'name'])

    for i,train_path in enumerate(train_paths):
        name = train_path.split("\\")[-1]
        images = glob.glob(train_path + "/*")
        for image in images:
            df_train.loc[len(df_train)]=[image,i,name]
            
    label2idx = []

    for i in range(len(train_paths)):
        label2idx.append(np.asarray(df_train[df_train.label == i].index))

    train_embs = np.load("train_embs.npy", allow_pickle=True)
    train_embs = np.concatenate(train_embs)

    return train_embs, label2idx, nb_classes, df_train

def calc_emb_vector(alignment, emb_model, face):
    aligned = align_face(alignment, face)
    aligned = (aligned / 255.).astype(np.float32)
    aligned = np.expand_dims(aligned, axis=0)

    emb_vec = emb_model.predict(aligned)
    return np.array(emb_vec)

def find_nearlest_face(emb_vec, train_embs, label2idx, nb_classes):

    distances = []
    for j in range(nb_classes):
        distances.append(np.min([distance.euclidean(emb_vec.reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
    return np.argsort(distances)[:1][0], np.min(distances)

if __name__ == "__main__":

    # stream_url = 'rtsp://admin:MOlang1992@192.168.1.125:554/cam/realmonitor?channel=1&subtype=0'
    stream_url = 'videos/2.mp4'

    capture = cv2.VideoCapture(stream_url)

    # Define the codec and create VideoWriter object
    w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if save_output_video:
        # video writer:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        writer = cv2.VideoWriter('videos/output.mp4', fourcc, 24.0, (int(w),int(h)))

    if recognize_face:
        alignment = AlignDlib('pre_models/shape_predictor_68_face_landmarks.dat')
        emb_model = create_model()
        emb_model.load_weights('weights/nn4.small2.v1.h5', by_name=True)
        train_embs, label2idx, nb_classes, df_train = load_data_embs()

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    frame_count = 0

    time_begin = time.time()
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        frame_copy = frame.copy()

        # detect faces:
        frame_copy = cv2.resize(frame_copy, (int(w/scale_factor), int(h/scale_factor)))
        faceRects, time_detect = detect_faces(frame_copy, scale_factor)
        # if len(faceRects) > 0:
            # print('detect {} faces on frame, time elapsed: {} second'.format(len(faceRects), time_detect))
        
        time_recognize = 0
        for i,faceRect in enumerate(faceRects):
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()

            # save to dataset:
            if save_faces:
                face = frame[y1:y2,x1:x2]
                cv2.imwrite('{}/IMG_{}_{}.jpg'.format(out_save_dir, frame_count, i), face)

            if recognize_face:
                time_start = time.time()
                face = frame[y1:y2,x1:x2]
                emb_vec = calc_emb_vector(alignment, emb_model, face)
                class_idx, min_distance = find_nearlest_face(emb_vec, train_embs, label2idx, nb_classes)
                time_recognize += time.time()-time_start
                # print('face recognized, time elapsed: {} second'.format(time_recognize))

                if min_distance > threshold:
                    class_name = "unknown"
                    cv2.imwrite('{}/UNKNOWN_{}_{}.jpg'.format(out_save_dir, frame_count, i), face)
                else:
                    class_name = df_train[(df_train['label']==class_idx)].name.iloc[0]
            # mark to render:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            if recognize_face:
                cv2.putText(frame,'{} ({:.04f})'.format(class_name, min_distance),(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1,cv2.LINE_AA)
    
        frame_count+=1
        cv2.putText(frame,'Face (time detect/recogn): {} ({:.04f}/{:.04f} s)'.format(len(faceRects), time_detect, time_recognize),(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1,cv2.LINE_AA)
        cv2.imshow("video",frame)
        
        # save to video output file: 
        if save_output_video: 
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    if save_output_video:   
        writer.release()
    print('Process completed, time elapsed: {} seconds'.format(time.time() - time_begin))