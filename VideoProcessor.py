from Tracker import Tracker
import cv2
from Detector import Detector

class TrackHistory:
	def __init__(self, track):
		self.id = track.id
		self.detections_dict = {}
		self.last_frame = track.starting_frame - 1
		for det in track.detections:
			self.last_frame += 1
			if det != 'missing':
				self.detections_dict.update({self.last_frame: det})
	
	def update(self, track):
		assert self.id == track.id, 'Track has different id.'
		self.last_frame += 1
		last_detection = track.detections[len(track.detections) - 1]
		if last_detection != 'missing':
			self.detections_dict.update({self.last_frame: last_detection})

class VideoProcessor:
	def __init__(self, keep_history=False):
		self.keep_history = keep_history
		self.tracker = Tracker()
		self.tracks_history = []
		self.frames = []
		self.detections = []

	def initialize(self):
		#self.detector = Detector('YOLO')
		self.detector = Detector('SSD-mobilenet-face')
		self.detector.initialize()

	def process_frame(self, frame):
		print('')
		# Detect in the current frame:
		self.image = frame
		self.detections = self.detector.detect(frame)
		# Update tracker:
		self.tracker.update_tracks(self.detections)
		# Update history:
		if self.keep_history:
			self.frames.append(frame)
			self.update_history()

	def get_last_frame_with_detections(self):
		img_cp = self.image.copy()
		print(str(len(self.tracker.get_tracks())) + ' tracks')
		for tr in self.tracker.get_tracks():
			detection = tr.get_last_detection()
			x = int(detection.x_center)
			y = int(detection.y_center)
			w = int(detection.width)//2
			h = int(detection.height)//2
			if tr.get_nframes_missing() > 0:
				print('missing frame for track ' + str(tr.get_id()) + ' - nframes_missing: ' + str(tr.get_nframes_missing()))
				color = (255, 0, 0)
			else:
				color = (0, 255, 0)
			cv2.rectangle(img_cp, (x-w,y-h), (x+w,y+h), color, 2)
			cv2.rectangle(img_cp, (x-w,y-h-20), (x+w,y-h), (125,125,125), -1)
			cv2.putText(img_cp, detection.class_name + ' ' + str(tr.get_id()) + ' : %.2f' % detection.conf, (x-w+5,y-h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
		return img_cp

	def clear(self):
		self.tracker.clear()
		self.tracks_history = []
		self.frames = []
		self.detections = []

	def update_history(self):
		for track in self.tracker.tracks:
			exists = False
			for i in range(len(self.tracks_history)):
				if self.tracks_history[i].id == track.id:
					exists = True
					self.tracks_history[i].update(track)
					break
			if not exists:
				self.tracks_history.append(TrackHistory(track))
