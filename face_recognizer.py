import sys
sys.path.append('/home/xian/brainlab')
import BrainLabNet
from config.predict_config_faces_classifier import UpdatePredictConfiguration
import numpy as np
import utils
import copy
import tensorflow as tf


class FaceRecognizer:
    def __init__(self):
        self.args = UpdatePredictConfiguration()
        g2 = tf.Graph()
        with g2.as_default() as g:
            # with g.name_scope('g2') as g2_scope:
            print('vamos a crear a rede')
            self.net = BrainLabNet.BrainLabNet(self.args, 'interactive')
            print('rede creada')
            print('vamos a comezar a sesion')
            self.net.start_interactive_session(self.args)
            print('sesion comezada')

    def get_person_name(self, image, detection):
        crop = utils.crop_image_from_detection_with_square(image, detection)
        crop = self.net.reader.preprocess_image(crop)
        crop = np.expand_dims(crop, axis=0)
        predictions = self.net.forward_batch(crop, self.args)
        winner = self.net.classnames[np.argmax(predictions[0])]
        return winner

    def get_person_name_enlarging(self, image, detection):
        detection_enlarged = copy.deepcopy(detection)
        detection_enlarged.width = 2 * detection.width
        detection_enlarged.height = 2 * detection.height
        crop = utils.crop_image_from_detection_with_square(image, detection_enlarged)
        crop = self.net.reader.preprocess_image(crop)
        crop = np.expand_dims(crop, axis=0)
        predictions = self.net.forward_batch(crop, self.args)
        winner = self.net.classnames[np.argmax(predictions[0])]
        return winner



