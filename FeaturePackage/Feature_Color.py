# B0843020 You-Yu 
'''
Todo:
This file is one of a feature extract processing function,it will read the input .jpg file, then output the
number of k's color  which are the major color in the .jpg file.
處理過程:先讀取圖片後,以凸包的方式作去背的動作,最後採用K-Means演算法計算k個主要的顏色分布,輸出各種主要的數據,並且用圓餅圖呈現出來

'''

from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
import os

#rgb to hex
def removeblack(the_list,val):
    '''
    回傳非黑區域
    '''
    return [value for value in the_list if value != val]

def rgb2hex(rgb):
    '''
    將rgb的每層轉換成2位的16進制表示式,使所有顏色有不同的色碼
    '''
    hex = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return hex

#k=分群數
def plot_image_info(path,img):
    '''
    將圖片做前處理,進行去背及凸包,並回傳去掉背景(to黑色),留下糞便輪廓的bgr之cv2img的array形式
    '''
    # load image
    img_bgr = img
    img_bgr_copy = img_bgr.copy()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_coffee = np.array([0,84,25])
    upper_coffee = np.array([66,255,150]) #咖啡色hsv
    mask = cv2.inRange(hsv, lower_coffee, upper_coffee)
    #橢圓形遮罩
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    #膨脹再腐蝕,更加邊緣去背效果
    mask = cv2.dilate(mask ,kernel1)
    mask = cv2.erode(mask, kernel2)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    #膨脹再腐蝕,更加邊緣去背效果
    mask = cv2.dilate(mask ,kernel3)
    mask = cv2.erode(mask, kernel4)
    out = cv2.bitwise_and(img_bgr, img_bgr, mask= mask)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) # 灰階
    blurred = cv2.GaussianBlur(gray, (11,11), 0)
    canny = cv2.Canny(blurred, 1, 80)
    ret, thresh = cv2.threshold(canny, 50, 255, cv2.THRESH_BINARY)

    # 找輪廓
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img_bgr, contours, -1, (255,0,255), 3)

    hull = []

    #先計算每個點
    for i in range(len(contours)):
        #創建凸包
        hull.append(cv2.convexHull(contours[i], False))

    #背景先用0
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # 畫圖
    for i in range(len(contours)):
        color_contours = (0, 255, 0) 
        color = (255, 255, 255)
        # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        cv2.drawContours(drawing, hull, i, color, -1, 8)

    #與原圖取交集
    imgbgr_drawing_and =cv2.bitwise_and(img_bgr_copy, drawing, dst=None, mask=None)

    # 去背完後的圖片顯示
    # cv2.imshow("black",imgbgr_drawing_and)
    # cv2.waitKey()
    return imgbgr_drawing_and

def plot_image_Kmeans(imgbgr_drawing_and,k):
    '''
    將cv2img的bgr照片當作輸入,裡用KMeans演算法輸出主要的k個顏色結果
    '''
    img_rgb = cv2.cvtColor(imgbgr_drawing_and, cv2.COLOR_BGR2RGB)
    # resized_img_rgb = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    resized_img_rgb = img_rgb
    # 換成一維
    img_list = resized_img_rgb.reshape((resized_img_rgb.shape[0] * resized_img_rgb.shape[1], 3))
    img_real=list()
    for pixel in img_list:
        if pixel[0]==0 and pixel[1]==0 and pixel[2]==0:
            doNothing=1 #nothing
        else:
            img_real.append(pixel)  #這邊寫法不好,要再改

    # 直接用sklearn.cluster的K-Means分群演算法
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(img_real)

    # 對每個k計數
    label_counts = Counter(labels)
    total_count = sum(label_counts.values()) # 500*500=250000
    center_colors = list(clt.cluster_centers_)
    ordered_colors = [center_colors[i]/255 for i in label_counts.keys()]
    
    color_labels = [rgb2hex(ordered_colors[i]*255) for i in label_counts.keys()]
    
    # print(label_counts.values())
    # print(color_labels)
    return color_labels
    
    # 畫圖
    # plt.figure(figsize=(14, 8))
    # plt.subplot(221)
    # plt.imshow(img_rgb)
    # plt.axis('off')
    
    # plt.subplot(222)
    # plt.pie(label_counts.values(), labels=color_labels, colors=ordered_colors, startangle=90)
    # plt.axis('equal')
    # plt.show()

def featureMainColor(Path,img):
    '''
    將cv2img的bgr照片當作輸入,計算糞便中5大顏色(黃、綠、紅、藍、白、其他)及其他色所佔的比例(以百分比計算)
    '''
    imgbgr_drawing_and = plot_image_info(Path,img)
    hsv = cv2.cvtColor(imgbgr_drawing_and,cv2.COLOR_BGR2HSV)
    # colorListYGRBWO內的元素 = [yellow,green,red,black,white,others] 之計數
    colorListYGRBWO=[0,0,0,0,0,0]
    pixelLen = len(hsv)*len(hsv[0])
    # OpenCV 使用HSV時的範圍 H: 0-179, S: 0-255, V: 0-255
    # v<90 is black
    # s<70 && v>180 is white
    # h>170 || h<10 is red
    # 10<h<35 is yellow
    # 35<h<75 is green
    # else is others
    for i in hsv:
        for hsvValue in i:
            if hsvValue[2]<5:
                pixelLen-=1        #except background
            elif hsvValue[2]<90:
                colorListYGRBWO[3]+=1 #black++
            elif hsvValue[1]<70 and hsvValue[2]>180:
                colorListYGRBWO[4]+=1 #white++
            elif hsvValue[0]>=170 or hsvValue[0]<=10:
                colorListYGRBWO[2]+=1 #red++
            elif hsvValue[0]>=75:
                colorListYGRBWO[5]+=1 #others++
            elif hsvValue[0]>=35:
                colorListYGRBWO[1]+=1 #green++
            elif hsvValue[0]>10:
                colorListYGRBWO[0]+=1 #yellow++
    colorListYGRBWO = [round(i/pixelLen,2) for i in colorListYGRBWO]
    return colorListYGRBWO

#main
def featureColor(Path,img):
    color=[]
    for path in Path:
        # print(path)
        imgbgr_drawing_and=plot_image_info(path,img)
        color.append(plot_image_Kmeans(imgbgr_drawing_and,6))
    return color

