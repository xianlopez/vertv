import cv2

def draw_detections(image, all_detections):
	img_cp = image.copy()
	for detection in all_detections:
		x = int(detection.x_center)
		y = int(detection.y_center)
		w = int(detection.width) // 2
		h = int(detection.height) // 2
		color = (0, 255, 0)
		cv2.rectangle(img_cp, (x-w,y-h), (x+w,y+h), color, 2)
		cv2.rectangle(img_cp, (x-w,y-h-20), (x+w,y-h), (125,125,125), -1)
		cv2.putText(img_cp, detection.class_name + ' ' + ' : %.2f' % detection.conf, (x-w+5,y-h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
	return img_cp

def crop_image_from_detection_with_square(image, detection):
	max_side = max(detection.width, detection.height)
	xmin = detection.x_center - max_side // 2
	ymin = detection.y_center - max_side // 2
	xmax = detection.x_center + max_side - max_side // 2
	ymax = detection.y_center + max_side - max_side // 2
	return crop_image(image, xmin, ymin, xmax, ymax)

def crop_image_from_detection(image, detection):
	xmin = detection.x_center - detection.width // 2
	ymin = detection.y_center - detection.height // 2
	xmax = detection.x_center + detection.width - detection.width // 2
	ymax = detection.y_center + detection.height - detection.height // 2
	return crop_image(image, xmin, ymin, xmax, ymax)

def crop_image(image, xmin, ymin, xmax, ymax, pad=True):
	height, width, _ = image.shape
	x0 = int(max(xmin, 0))
	y0 = int(max(ymin, 0))
	x1 = int(min(xmax, width))
	y1 = int(min(ymax, height))
	crop = image[y0:y1, x0:x1]
	if pad:
		top = int(y0 - ymin)
		bottom = int(ymax - y1)
		left = int(x0 - xmin)
		right = int(xmax - x1)
		crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)
	return crop
