from VideoProcessor import VideoProcessor
import os
import cv2

fps = 30
videoPath = r'/home/xian/vertv/videos/pabloiglesias.mp4'

videoProcessor = VideoProcessor()
videoProcessor.initialize()

cap = cv2.VideoCapture(videoPath)

while True:
    # Capture frame-by-frame:
    ret, frame = cap.read()
    if not ret:
        break
    videoProcessor.process_frame(frame)
    frame_w_boxes = videoProcessor.get_last_frame_with_detections()
    cv2.imshow('People detection', frame_w_boxes)
    cv2.waitKey(int(1000 / fps))





