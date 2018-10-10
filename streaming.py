import time
import numpy as np
import cv2
import sys
from YOLO_small_tf import YOLO_TF

#link = "http://hlsliveamdgl7-lh.akamaihd.net/i/hlsdvrlive_1@583042/master.m3u8" # la 1
link = "http://a3live-lh.akamaihd.net/i/lasexta/lasexta_1@35272/master.m3u8" # la sexta

cap = cv2.VideoCapture(link)
yolo = YOLO_TF(sys.argv)

while(True):
    #time.sleep()

    # Capture frame-by-frame
    ret, frame = cap.read()

#    cv2.imwrite('imaxe.png', frame)
    
#    argv = ['YOLO_small_tf.py', '-fromfile', 'imaxe.png']
#    yolo.detect_from_file('imaxe.png')
    yolo.detect_from_cvmat(frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
