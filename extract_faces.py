from Detector import Detector
import os
import cv2
import utils
import numpy as np

dirimgs = os.path.join('/home/xian/imaxes_politicos/Imaxes')
dircaras = os.path.join('/home/xian/imaxes_politicos/Caras')
factor = 2

detector = Detector('SSD-mobilenet-face', threshold=0.5)
detector.initialize()

people = os.listdir(dirimgs)

for person in people:
	imgs = os.listdir(os.path.join(dirimgs, person))
	for imgname in imgs:
		print(person + ' - ' + imgname)
		imaxe = cv2.imread(os.path.join(dirimgs, person, imgname))
		result = detector.detect(imaxe)
		if len(result) == 0:
			continue
		elif len(result) == 1:
			detection = result[0]
		else:
			conf = []
			for det in result:
				conf.append(det.conf)
			idx = np.argmax(np.float32(conf))
			detection = result[idx]
		detection.width *= factor
		detection.height *= factor
		face = utils.crop_image_from_detection_with_square(imaxe, detection)
		newname = os.path.join(dircaras, person, imgname)
		cv2.imwrite(newname, face)

#		if len(result) != 1:
#			img_caras = utils.draw_detections(imaxe, result)
#			cv2.imshow('caras', img_caras)
#			cv2.waitKey(0)
#			raise Exception('Image ' + imgname + ' has ' + str(len(result)))









