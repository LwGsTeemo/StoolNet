"""記錄下所有Yolov5辨識""異物特徵""後的資料
"""
import cv2
import glob
import os

def featureObjDetect(Path,img,expN):
    for i in Path:
        basename = os.path.basename(i).split('.')[0]
        objcount=0
        object=[]
        size = img.shape
        h = size[0]
        w = size[1]
        #取label.txt中的資料
        # image_ids = open(os.getcwd()+'\data\labels\\'+basename+'.txt').read().strip().split()
        image_ids = open(os.getcwd()+'\\runs\detect\exp'+str(expN)+'\labels\\'+basename+'.txt').read().strip().split()
        #將yolo抓出為'2'(異物)的資料計數
        for j in range(0,len(image_ids),5):
            x = float(image_ids[j+1])*w
            y = float(image_ids[j+2])*h
            hw = float(image_ids[j+3])*w*0.5
            hh = float(image_ids[j+4])*h*0.5
            if image_ids[j]=='0': #stool
                img = cv2.rectangle(img, (int(x-hw),int(y-hh)), (int(x+hw),int(y+hh)), (0, 200, 0), 2)
            if image_ids[j]=='1': #bubble
                img = cv2.rectangle(img, (int(x-hw),int(y-hh)), (int(x+hw),int(y+hh)), (0, 0, 150), 2)
            if image_ids[j]=='2': #strange
                objcount+=1
                object.append([image_ids[j],x,y,hw,hh])
                img = cv2.rectangle(img, (int(x-hw),int(y-hh)), (int(x+hw),int(y+hh)), (255, 0, 0), 2)
        
        return objcount
        # print(basename+'.jpg',end=' ')
        # print("異物數量:",objcount)
        # cv2.imshow('img',img)
        # cv2.waitKey()
