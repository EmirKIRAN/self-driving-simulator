from posixpath import basename
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.getcwd()
videos_dir = os.path.join(base_dir, 'test_videos')
video_names = os.listdir(videos_dir) #! test videolarının isimleri(liste)

capture = cv2.VideoCapture(os.path.join(videos_dir, video_names[0]))

while True:

    _, frame = capture.read()

    cv2.imshow('Screen', frame)

    if cv2.waitKey(50)&0xFF == ord('q'):
        break
