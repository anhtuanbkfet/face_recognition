import cv2
import dlib
import time

if __name__ == "__main__":


    run_mode = 'CAPTURE' #GET_DATASET

    # define face detector:
    hogFaceDetector = dlib.get_frontal_face_detector()

    stream_url = 'rtsp://admin:MOlang1992@192.168.1.125:554/cam/realmonitor?channel=1&subtype=0'

    capture = cv2.VideoCapture(stream_url)

    # Define the codec and create VideoWriter object
    w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    f_rate = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    file_name = 'video/captured_{}.avi'.format(int(time.time()))
    video_writer = cv2.VideoWriter(file_name, fourcc, f_rate, (int(w),int(h)))

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    out_save_dir = 'DATASET/UNKNOWN'

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        show_image = frame.copy()

        if run_mode == 'CAPTURE':
            video_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('f'):
                cv2.imwrite('test_image/IMG_{}.jpg'.format(frame_count//10), frame)
        else:
            if frame_count % 10 == 0:
                # detect faces:     
                faceRects = hogFaceDetector(frame, 0)

                # mark with green rectangles:
                for i,faceRect in enumerate(faceRects):
                    x1 = faceRect.left()
                    y1 = faceRect.top()
                    x2 = faceRect.right()
                    y2 = faceRect.bottom()
                    cv2.rectangle(show_image,(x1,y1),(x2,y2),(0,0,255),2)

                    # save to dataset:
                    crop = frame[y1:y2,x1:x2]
                    cv2.imwrite('{}/IMG_{}.jpg'.format(out_save_dir, frame_count//10), crop)
        
        frame_count+=1
        cv2.imshow("video",show_image)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    video_writer.release()