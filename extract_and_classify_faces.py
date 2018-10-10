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
import utils
import datetime

percent_frames = 5

prob_save = 1

videosdir = '/home/xian/imaxes_politicos/Videos'
facesdir = '/home/xian/imaxes_politicos/facesfromvideos'

face_count = 0

def process_tracks(tracks):
	for tr in tracks:
		nselect = max(int(math.round(percent_frames / 100.0 * len(tr.detections))), 0)
		selected_det = random.sample(tr.detections, nselect)
		for det in selected_det:
			face_count += 1
			classdir = os.path.join(facesdir, det.value.class_name)
			if not os.path.exists(classdir):
				os.makedirs(classdir)
			
			

videos_list = os.listdir(videosdir)

videoProcessor = VideoProcessor()
videoProcessor.initialize()

for videoname in videos_list:
	videopath = os.path.join(videosdir, videoname)
	cap = cv2.VideoCapture(videopath)

	while(cap.isOpened()):
		ret, frame = cap.read()
		if (type(frame) == type(None)):
		    break
		videoProcessor.process_frame(frame)
		frame_w_boxes = videoProcessor.get_last_frame_with_detections()

	    	detections_now = videoProcessor.detections
		for det in detections_now:
			if random.random() < prob_save / 100.0:
				face_count += 1
				classdir = os.path.join(facesdir, det.class_name)
				if not os.path.exists(classdir):
					os.makedirs(classdir)
				crop = utils.crop_image_from_detection_with_square(frame, det)
				img_name = 'face' + str(face_count) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
				cv2.imwrite(os.path.join(classdir, img_name), crop)
				

		cv2.imshow('frame', frame_w_boxes)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

#	tracks_found = videoProcessor.tracks_history
#	process_tracks(tracks_found)
	videoProcessor.clear()

	# When everything is done, release the capture
	cap.release()
	cv2.destroyAllWindows()






