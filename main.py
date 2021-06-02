from posixpath import basename
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.getcwd()
videos_dir = os.path.join(base_dir, 'test_videos')
video_names = os.listdir(videos_dir) #! test videolarının isimleri(liste)

def convert_RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def RGB_color_selection(image):

    img = convert_RGB(image)

    #! Şeritlerdeki beyaz rengi algılamak için alt ve üst değer belirlendi.
    lower_threshold = np.uint8([200,200,200])
    upper_threshold = np.uint8([255,255,255])
    white_mask = cv2.inRange(img, lower_threshold, upper_threshold) #* Beyaz renk için maske elde edildi.

    #! Şeritlerdeki sarı rengi algılamak için alt ve üst değer belirlendi
    lower_threshold = np.uint8([175,175,0])
    upper_threshold = np.uint8([255,255,255])
    yellow_mask = cv2.inRange(img, lower_threshold, upper_threshold)#* Sarı renk için maske elde edildi.

    #! beyaz ve sarı maskedeki görüntüler 'or' ile birleştirildi.
    mask = cv2.bitwise_or(white_mask, yellow_mask)  
    #! maske ana görselin üzerine uygulandı.
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    return masked_image

def convert_HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

capture = cv2.VideoCapture(os.path.join(videos_dir, video_names[0]))


while True:

    _, frame = capture.read()

    cv2.imshow('Screen', RGB_color_selection(frame))

    if cv2.waitKey(50)&0xFF == ord('q'):
        break
