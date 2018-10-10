from YOLO_small_tf import YOLO_TF
from face_detector import ssd_mobilenet_face_wrapper
from face_recognizer import FaceRecognizer


class Detection:
	class_name = ''
	x_center = -1
	y_center = -1
	width = -1
	height = -1
	conf = -1


class Detector:
	def __init__(self, network_name, threshold=0.7):
		self.network_name = network_name
		self.threshold = threshold

	def initialize(self):
		if self.network_name == 'YOLO':
			self.network = YOLO_TF()
		elif self.network_name == 'SSD-mobilenet-face':
			self.network = ssd_mobilenet_face_wrapper.ssd_mobilenet_face(self.threshold)
			self.faceRecognizer = FaceRecognizer()
		else:
			raise Exception('Network name not recognized')

	def detect(self, image):
		if self.network_name == 'YOLO':
			yolo_result = self.network.detect_from_cvmat2(image)
			all_detections = []
			for i in range(len(yolo_result)):
				if yolo_result[i][0] == 'person':
					this_detection = Detection()
					this_detection.class_name = yolo_result[i][0]
					this_detection.x_center = yolo_result[i][1]
					this_detection.y_center = yolo_result[i][2]
					this_detection.width = yolo_result[i][3]
					this_detection.height = yolo_result[i][4]
					this_detection.conf = yolo_result[i][5]
					all_detections.append(this_detection)

		elif self.network_name == 'SSD-mobilenet-face':
			boxes = self.network.forward_image(image)
			all_detections = []
			for i in range(len(boxes)):
				this_detection = Detection()
				this_detection.class_name = boxes[i][0]
				this_detection.x_center = boxes[i][1]
				this_detection.y_center = boxes[i][2]
				this_detection.width = boxes[i][3] * 2
				this_detection.height = boxes[i][4] * 2
				this_detection.conf = boxes[i][5]
				this_detection.class_name = self.faceRecognizer.get_person_name(image, this_detection)
				all_detections.append(this_detection)

		else:
			raise Exception('Network name not recognized')

		#print('Found ' + str(len(all_detections)) + ' detections')

		return all_detections













