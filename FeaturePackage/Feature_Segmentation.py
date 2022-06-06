"""記錄下所有Yolov5辨識""糞便特徵""後的資料
"""
import cv2
import glob
import os

#利用yolo分辨出每張圖片的糞便及其位置,再利用label.txt的內容將糞便計數
def featureSegmentation(Path,expN):
#每張照片的路徑
# Path = glob.glob(os.getcwd()+'\data\images\*.jpg')
    for i in Path:
        basename = os.path.basename(i).split('.')[0]
        stoolCnt=0
        #取label.txt中的資料
        # image_ids = open(os.getcwd()+'\data\labels\\'+basename+'.txt').read().strip().split()
        image_ids = open(os.getcwd()+'\\runs\detect\exp'+str(expN)+'\labels\\'+basename+'.txt').read().strip().split()

        for j in range(0,len(image_ids),5):
            if image_ids[j]=='0': #stool
                stoolCnt+=1
                """
                新程式......
                """
        return stoolCnt

        #summary結果
        # print(basename+'.jpg',end=' ')
        # print("糞便數量:",count)
        # cv2.imshow('img',img)
        # cv2.waitKey()