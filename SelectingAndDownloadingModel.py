#!/usr/bin/env python3
'''
Run this script using python3 because of the urllib script
'''

import os
import shutil
import urllib.request

import tarfile

# Select a model from `MODELS_CONFIG`.
# I chose ssd_mobilenet_v2 for this project, you could choose any
selected_model = 'faster_rcnn_resnet50'

#the distination folder where the model will be saved
#change this if you have a different working dir
DEST_DIR = './models/pretrained_model/'


# Name of the object detection model to use.
MODEL = 'faster_rcnn_resnet50_lowproposals_coco_2018_01_28'

#selecting the model
MODEL_FILE = MODEL + '.tar.gz'

#creating the downlaod link for the model selected
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

#checks if the model has already been downloaded, download it otherwise
if not (os.path.exists(DEST_DIR + MODEL_FILE)):
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, DEST_DIR + MODEL_FILE)

#unzipping the model and extracting its content
tar = tarfile.open(DEST_DIR + MODEL_FILE)
tar.extractall()
tar.close()

# creating an output file to save the model while training
os.remove(DEST_DIR + MODEL_FILE)
if (os.path.exists(DEST_DIR)):
    shutil.rmtree(DEST_DIR)
os.rename(MODEL, DEST_DIR)