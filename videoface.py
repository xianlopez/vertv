import sys

sys.path.append('/home/xian/brainlab')

import BrainLabNet
from config.predict_config_face import PredictConfiguration
import os
import cv2
import tools
import random
import numpy as np
from VideoProcessor import VideoProcessor

videopath = '/home/xian/vertv/videos/pabloiglesias.mp4'
#videopath = '/home/xian/vertv/videos/rajoy.mp4'

videoProcessor = VideoProcessor()
videoProcessor.initialize()

cap = cv2.VideoCapture(videopath)

while(cap.isOpened()):
	ret, frame = cap.read()
	videoProcessor.process_frame(frame)
	frame_w_boxes = videoProcessor.get_last_frame_with_detections()

    

	cv2.imshow('frame', frame_w_boxes)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

