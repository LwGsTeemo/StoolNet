'''
常用的function會記錄在這
目前有:getPath,getImg,getCanny,getFile,getHealthPoint,getHealthResult
'''
import os,glob
import cv2
def getPath():
    '''
    獲取該張圖片位置
    '''
    Path = glob.glob(os.getcwd()+'\data\images\\1-0.jpg')
    return Path

def getImg(Path):
    '''
    讀取每個.jpg檔案,並採用cv2的插植法,將每一張照片做規一化(化成500*500*3)的照片處理
    '''
    for i in Path:
        img = cv2.imread(i)
    #resize成500x500,用cv2的插值法(放大採用cv2.INTER_CUBIC,縮小採用cv2.INTER_AREA 結果上效果最好)
        if img.shape[0]<500:
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("pic",img)
        # cv2.waitKey()
        # return第一張照片
        return img

def getCanny(Path):
    '''
    針對照片進行Canny演算法,並顯示
    '''
    for i in Path:
        print(i)
        image = cv2.imread(i)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blurred, 1, 80)
        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        print(cnts)
        # cv2.imshow("beforeImage",image)
        # cv2.imshow("image",canny)
        cv2.waitKey(0)

def getFile(folderpath,filepath,originalformat,newformat):
    '''
    此函式為簡單的轉檔、改圖大小,可將任意副檔名轉成jpg或其他欲轉換的格式
    originalformat為原本格式
    newformat為欲轉換的格式
    更改這兩項變數就可以了,路徑注意一下就行
    格式範例:
    folderpath = 'D:\ConSinGAN-master\TrainedModels\\3-3\\2022_01_23_04_40_16_generation_train_depth_3_lr_scale_0.1_act_lrelu_0.05\gen_samples_stage_4\\'
    originalformat = str('HCIE')
    newformat = str('jpg')
    filepath = glob.glob(folderpath + '*.' + str(originalformat)) 
    '''
    print(filepath)
    SLfilelist = [] 
    n = 0
    for i in filepath:
        SLfilelist.append(os.path.splitext(os.path.basename(filepath[n]))[0]) 
        # SLfilelist.append(os.path.splitext(os.path.basename(filepath[n]))[0]) 
        n+=1
    n = 0
    for i in SLfilelist:
        oldpath = folderpath + SLfilelist[n] + '.' + str(originalformat)  
        newpath = folderpath + SLfilelist[n] + '.' + str(newformat)
        os.rename(oldpath, newpath) 
        n+=1
    print("完成")

def getHealthPoint(featureDic):
    '''
    將擷取完的特徵透過文獻算式,輸出成一個健康度分數,0代表最低,10代表最健康,精準度為.1f
    '''
    healthPoint=0.0
    colorYGRBWO,blood,bss,bubble = featureDic['color'],featureDic['blood'],featureDic['bss'],featureDic['bubble']

    #colorYGRBWO [0.20,0.20,0.20,0.20,0.20] [0]>0.7:+2 
    if colorYGRBWO[0]>=0.6:healthPoint+=2
    elif colorYGRBWO[0]>=0.5 and colorYGRBWO[1]>=0.1:healthPoint+=1.5
    elif colorYGRBWO[0]>=0.5 and colorYGRBWO[2]<=0.08:healthPoint+=1.5
    elif colorYGRBWO[3]<=0.4 and colorYGRBWO[2]<=0.08:healthPoint+=1.5

    #blood 0 1 0:+2.5  1:+0
    if blood==1:healthPoint+=2.5

    #bss 1 2 3 4 5 6 7   3、4:+2.5 2、5:+1.5 6:+1 1、7:+0
    if bss ==3 or bss==4:healthPoint+=2.5
    elif bss ==2 or bss==5:healthPoint+=1.5
    elif bss ==6:healthPoint+=1

    #bubble 0~100   0~30:+2.5  30~60:+1.5 60~100:+0
    if bubble<=30:healthPoint+=2.5
    elif bubble<=60:healthPoint+=1.5

    #healthPointSum 1~10 良:7+up 中:4~6 惡:0~3
    
    #考慮到未來擴充性，健康分數應該要將每一種特徵的分數去做加權
    return healthPoint

def getHealthResult(featureDic,sep):
    '''
    以回傳文字(string)來顯示出各種特徵的結果,並以sep(ex:' ')分別隔開
    '''
    colorYGRBWO,blood,bss,obj,bubble= featureDic['color'],featureDic['blood'],featureDic['bss'],
    featureDic['obj'],featureDic['bubble']

    healthResult=""

    # for debugging:
    # colorYGRBWO,stoolCnt,blood,bss,obj,bubble=[0.58, 0.0, 0.0, 0.42, 0.0, 0.0],2, 0, 5, 1, 36.15
    YGRBWO =["黃色","綠色","紅色","黑色","白色","其他"]
    healthResult+="主要顏色為"+YGRBWO[colorYGRBWO.index(max(colorYGRBWO))]+sep # ex:'主要顏色為黃色'

    blood="有" if blood else "沒有"
    healthResult+=blood+"辨識出血液"+sep # ex:'沒有辨識出血液'

    bss ="腹瀉型" if bss>4 else "正常型" if bss>2 else "便祕型"
    healthResult+="外型為"+bss+sep # ex:'外型為腹瀉型'

    obj="有" if obj else "沒有"
    healthResult+="馬桶周圍"+obj+"辨識出異物"+sep #ex:'馬桶周圍有辨識出異物'

    bubble="過多" if bubble>=60 else "偏多" if bubble>=40 else "正常"
    healthResult+="糞便周圍氣泡數量"+bubble+sep #ex:'糞便周圍氣泡數量正常'

    healthResult+="若糞便長期有以下 便祕型、腹瀉型、有血液、氣泡過多 之任一種，建議就醫!"

    return healthResult #type==string
