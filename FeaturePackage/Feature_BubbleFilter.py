'''(舊)該程式的功能為看該患者是否腸胃有問題(input:圖片,output:0(沒問題) or 1(有問題) )
若該患者有問題，可顯示出腸胃問題的嚴重程度(1~100)
'''

import cv2
import os
# from scipy.spatial import distance as dist
import copy
import numpy as np

# 讀入每張照片的長寬 跟label.txt的中心比 長寬長度比 反求四個端點位置 再和每個泡泡的中心點和糞便的邊緣輪廓位置取距離
# 至於test的時候input是jpg output是txt 所以之後找每張照片的長跟寬不是從xml找 要利用opencv直接讀jpg的長寬

def featureBubbleDetect(Path,img,expN):
    stool=list()
    bubble=list()
    for i in Path:
        basename = os.path.basename(i).split('.')[0]
        count=0
        size = img.shape
        h = size[0]
        w = size[1]
        #BGR轉hsv,先將糞便的相關顏色塞選出來
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
        out = cv2.bitwise_and(img, img, mask= mask)
        #canny檢測邊緣並抓出輪廓
        gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        # for j in range(h):
        #     for k in range(w):
        #         if (int(gray[j, k] * 1.5) > 255):
        #             grey = 255
        #         else:
        #             grey = int(gray[j, k] * 1.5)
        #         gray[j, k] = np.uint8(grey)
        #以上為灰值度增強1.5倍
        blurred = cv2.GaussianBlur(gray, (11,11), 0)
        canny = cv2.Canny(blurred, 1, 80)
        cnts,hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] #if len(cnts) == 2 else cnts[1]
        img = cv2.drawContours(img, cnts, -1, (255,0,255), 3)
        afterimg = img.copy()
        #讀取label.txt
        # image_ids = open(os.getcwd()+'\data\labels\\'+basename+'.txt').read().strip().split()
        image_ids = open(os.getcwd()+'\\runs\detect\exp'+str(expN)+'\labels\\'+basename+'.txt').read().strip().split()
        stool=[]
        bubble=[]
        object=[]
        bubbleLength=[]
        bubbleArea=[]
        #抓出每個.txt中每一行的中心比 長寬長度比
        for j in range(0,len(image_ids),5):
            # 分段5個一數，將糞便跟氣泡分開來存放成list 整理好數值後再算距離
            x = float(image_ids[j+1])*w
            y = float(image_ids[j+2])*h
            hw = float(image_ids[j+3])*w*0.5
            hh = float(image_ids[j+4])*h*0.5
            if image_ids[j]=='0': #stool
                stool.append([image_ids[j],x,y,hw,hh])
            if image_ids[j]=='1': #bubble
                bubble.append([image_ids[j],x,y,hw,hh])
                #先畫出原本yolov5所框出的泡泡位置,拿來之後比較
                img = cv2.rectangle(img, (int(x-hw),int(y-hh)), (int(x+hw),int(y+hh)), (0, 255, 0), 2)
        for k in bubble:
            if k:
                for c in cnts:
                    #點到輪廓的距離
                    x1 = cv2.pointPolygonTest(c,(k[1],k[2]), True)
                    #判斷距離,有的話就將該泡泡框出來
                    if x1<20 and x1>=-10:
                        # print(k) 這張圖的每個氣泡的位置
                        afterimg = cv2.rectangle(afterimg, (int(k[1]-k[3]),int(k[2]-k[4])), (int(k[1]+k[3]),int(k[2]+k[4])), (0, 255, 0), 2)
                        xbubble=(k[1]+k[3])-(k[1]-k[3])
                        ybubble=(k[2]+k[4])-(k[2]-k[4])
                        bubbleArea.append(float(xbubble*ybubble))
                        bubbleLength.append(float((xbubble+ybubble)*2))
                        count+=1
                        break
        #count為糞便周圍氣泡的個數,大於某個數量後就代表該張患者腸胃有問題
        if count >=1:
            #1.算出整個輪廓的距離，以及每個氣泡的周長總和   2.算出整個糞便輪廓的面積，以及每個氣泡的面積，再去做比例
            # 輪廓距離
            # print(i,"STATE 1")
            sumContoursLength=0
            sumBubbleLength=0
            sumContoursArea=0
            sumBubbleArea=0
            for j in cnts:
                sumContoursLength+=cv2.arcLength(j, False)
                sumContoursArea+=cv2.contourArea(j)
                # print(cv2.contourArea(j))
            #氣泡周長
            for j in bubbleLength:
                sumBubbleLength+=j
            # print(bubbleArea)
            # print(sumContoursArea)
            for j in bubbleArea:
                sumBubbleArea+=j
            # print("長度嚴重程度:",100*sumBubbleLength/sumContoursLength,"%")
            # print("面積嚴重程度:",100*sumBubbleArea/sumContoursArea,"%")
            return round(100*sumBubbleArea/sumContoursArea,2)
        else:
            # print(i,"STATE 0")
            # print("嚴重程度:0")
            return 0
        
        # #show出圖片,比較差異
        # cv2.imshow('img',img)
        # cv2.imshow('afterimg',afterimg)
        # cv2.waitKey()
