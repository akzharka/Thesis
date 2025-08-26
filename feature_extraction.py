import numpy as np
import ffmpeg
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input
from subprocess import call


class FeatureExtraction:
    def __init__(self, model, frame_number, src, dest):
        self.model = model
        self.frame_number = frame_number
        self.src = src
        self.dest = dest
        if model == 'vgg16':
            vgg16 = VGG16(weights='imagenet', include_top=True)
            self.backbone = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
            self.target_size = (224, 224)
        elif model == 'vgg19':
            vgg19 = VGG19(weights='imagenet', include_top=True)
            self.backbone = Model(inputs=vgg19.input,
                                  outputs=vgg19.get_layer('fc2').output)
            self.target_size = (224, 224)

    def extract_frames(self):
        # get information about input video
        probe = ffmpeg.probe(self.src)
        # extract duration of the video
        duration = float(probe['streams'][0]['duration'])
        # calculate new fps
        new_fps = str(self.frame_number / duration)
        # extract only required number of frames and save them in a separate folder
        call(["ffmpeg", "-i", self.src, "-filter:v", "fps=" + new_fps, self.dest + "/%03d.jpeg"])

    def extract_features(self, frame_path):
        frame = image.load_img(frame_path, target_size=self.target_size)
        x = image.img_to_array(frame)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.backbone.predict(x)
        features = features[0]
        return features






