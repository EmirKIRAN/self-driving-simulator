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
    
def get_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image):

    K_SIZE = 13
    #! İlgili görselleri blur vererek kenar tespitine daha uygun hale getiriyoruz.
    gaussian_image = cv2.GaussianBlur(image, (K_SIZE, K_SIZE), 0) 
    return gaussian_image

def edge_detector(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def select_region(image):

    _mask = np.zeros_like(image)
    #! giriş görüntüsüne bağlı olarak maskeyi doldurmak için 3 kanal veya 1 kanal rengi tanımla
    if len(image.shape) > 2:
        num_of_channel = image.shape[2]
        ignore_mask_color = (255, ) * num_of_channel
    else:
        ignore_mask_color = 255

    #! Aşağıdaki alanda çokgenin köşeleri olarak sabit sayıları kullanabildik.
    #! Ancak farklı boyutlara sahip resimler için geçerli olmayacaktır.

    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]

    vertices = np.array([[bottom_left, top_left, bottom_right, top_right]], dtype=np.int32)
    cv2.fillPoly(_mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, _mask)
    return masked_image

def hough_transform(image):
    rho = 1
    theta = np.pi/180
    threshold = 20
    minLineLength = 20
    maxLineGap = 300
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)

capture = cv2.VideoCapture(os.path.join(videos_dir, video_names[0]))


while True:

    _, frame = capture.read()

    #cv2.imshow('Screen 1', RGB_color_selection(frame))
    #cv2.imshow('Screen 2', HSV_color_selection(frame))
    hls_image = HLS_color_selection(frame)
    
    #! En uygun renk uzayının HSL olduğunu belirledik. Artık diğer işlemlere bu renk uzayı ile devam edeceğiz.
    #! Dilerseniz diğer renk uzaylarını da deneyerek aradaki farkı gözlemleyebilirsiniz.
    
    grayscale_image = get_gray_scale(hls_image) #! Görseli gray-scale formatına çevirerek diğer işlemleri yapabilmeyi sağladık.
    blurring_image = apply_smoothing(grayscale_image)
    edges = edge_detector(blurring_image)
    """
    ! Canny kenar bulma algoritmasını kullandıktan sonra elde ettiğimiz sonuç oldukça başarıldır.
    * Fakat ilk olarak amacımız şerit tespit ve takibi olduğu için araba, tabela ve ağaç gibi diğer objeleri ekarte etmeliyiz.
    * Bunun için her karede belirli alanları alarak işlemlere oradan devam etmeliyiz.
    """
    masked = select_region(edges)
    hough_lines = hough_transform(masked)
    #! hough transform fonksiyonu ile elde edilen cizgiler resim uzerinde cizdiriliyor.
    

    cv2.imshow('Screen 1', masked)
    cv2.imshow('Screen 2', edges)

    if cv2.waitKey(50)&0xFF == ord('q'):
        break
