import sys

sys.path.append('/home/xian/brainlab')

import BrainLabNet
from config.predict_config_face import PredictConfiguration
import os
import cv2
import tools
import random
import numpy as np



args = PredictConfiguration()

net = BrainLabNet.BrainLabNet(args, 'interactive')

net.start_interactive_session(args)

dir_imgs = '/home/xian/datasets/politicos/Imaxes'
people = os.listdir(dir_imgs)

for i in range(20):
	person = random.choice(people)
	imgs = os.listdir(os.path.join(dir_imgs, person))
	img_name = random.choice(imgs)
	img_path = os.path.join(dir_imgs, person, img_name)
	input_batch = net.reader.get_batch([img_path])
	predictions = net.forward_batch(input_batch, args)
	winner = net.classnames[np.argmax(predictions[0])]
	imaxe = cv2.imread(img_path)
	cv2.imshow(winner, imaxe)
	cv2.waitKey(0)
	cv2.destroyAllWindows()





