from VideoProcessor import VideoProcessor
import os
import cv2

videoProcessor = VideoProcessor()
videoProcessor.initialize()

base_dir = os.path.dirname(os.path.realpath(__file__))
frames_dir = os.path.join(base_dir, 'some_frames')
#frames_dir = os.path.join(base_dir, 'frames')

frames_list = os.listdir(frames_dir)
frames_list.sort()


for frame_name in frames_list:
    frame = cv2.imread(os.path.join(frames_dir, frame_name))
    videoProcessor.process_frame(frame)
    frame_w_boxes = videoProcessor.get_last_frame_with_detections()
    cv2.imshow('YOLO_small detection', frame_w_boxes)
    cv2.waitKey(int(1000 / 25))
    #cv2.waitKey()




