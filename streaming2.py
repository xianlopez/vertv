import time
import numpy as np
import cv2
import sys
from VideoProcessor import VideoProcessor

#link = "http://hlsliveamdgl7-lh.akamaihd.net/i/hlsdvrlive_1@583042/master.m3u8" # la 1
link = "http://a3live-lh.akamaihd.net/i/lasexta/lasexta_1@35272/master.m3u8" # la sexta

cap = cv2.VideoCapture(link)
videoProcessor = VideoProcessor()
videoProcessor.initialize()

while(True):
    #time.sleep()

    # Capture frame-by-frame
    ret, frame = cap.read()

#    cv2.imwrite('imaxe.png', frame)
    
#    argv = ['YOLO_small_tf.py', '-fromfile', 'imaxe.png']
#    yolo.detect_from_file('imaxe.png')
    videoProcessor.process_frame(frame)
    frame_w_boxes = videoProcessor.get_last_frame_with_detections()
    cv2.imshow('YOLO_small detection', frame_w_boxes)
    cv2.waitKey(int(1000 / 25))



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
