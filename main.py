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

def HSV_color_selection(image):

    img = convert_HSV(image)

    #! HSV renk uzayında beyaz şeritleri tespit edebilmek için alt ve üst değerleri belirliyoruz.
    lower_threshold = np.uint8([0,0,210])
    upper_threshold = np.uint8([255,30,255])
    white_mask = cv2.inRange(img, lower_threshold, upper_threshold)

    #! HSV renk uzayında sarı şeritleri tespit edebilmek için alt ve üst değerleri belirliyoruz.
    lower_threshold = np.uint8([18,80,80])
    upper_threshold = np.uint8([30,255,255])
    yellow_mask = cv2.inRange(img, lower_threshold, upper_threshold)

    #! beyaz ve sarı maskedeki görüntüler 'or' ile birleştirildi.
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    #! maske ana görselin üzerine uygulandı.
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    return masked_image

def convert_HLS(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def HLS_color_selection(image):

    img = convert_HLS(image)

    #! HLS renk uzayında beyaz renkte bulunan şeritleri tespit edebilmek için renk aralıklarını belirliyoruz.
    lower_thresholding = np.uint8([0,200,0])
    upper_thresholding = np.uint8([255,255,255])
    white_mask = cv2.inRange(img, lower_thresholding, upper_thresholding)

    #! HSL renk uzayında sarı renkte bulunan şeritleri tespit edebilmek için renk aralıklarını belirliyoruz.
    lower_thresholding = np.uint8([10,0,100])
    upper_thresholding = np.uint8([40,255,255])
    yellow_mask = cv2.inRange(img, lower_thresholding, upper_thresholding)

    #! Hem sarı hem de beyaz şeritleri aynı anda tanıyabilmek için iki maskeyi kombinliyoruz(or gate)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    #! Elde ettiğimiz maskeyi tüm görsele uygulayarak elde etmek istediğimiz görseli elde ediyoruz
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    return masked_image
    
capture = cv2.VideoCapture(os.path.join(videos_dir, video_names[0]))


while True:

    _, frame = capture.read()

    #cv2.imshow('Screen 1', RGB_color_selection(frame))
    #cv2.imshow('Screen 2', HSV_color_selection(frame))
    cv2.imshow('Screen 3', HLS_color_selection(frame))
    #! En uygun renk uzayının HSL olduğunu belirledik. Artık diğer işlemlere bu renk uzayı ile devam edeceğiz.
    #! Dilerseniz diğer renk uzaylarını da deneyerek aradaki farkı gözlemleyebilirsiniz.
    

    if cv2.waitKey(50)&0xFF == ord('q'):
        break
