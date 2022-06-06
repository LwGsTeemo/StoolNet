from FeaturePackage import Feature_Color
from FeaturePackage import Feature_Blood
from FeaturePackage import Feature_BSS
from FeaturePackage import Feature_BubbleFilter
from FeaturePackage import Feature_Object
from FeaturePackage import Feature_Segmentation
from FeaturePackage import Feature_Detect
from FeaturePackage import Feature_Function
import os
# Path:欲辨識的圖片的位置(以List儲存,可存復數張)
# img :經過正規化後的圖片(皆為(500*500*3))
# expN:因為會將每一次的辨識結果儲存下來,該變數為第expN次辨識結果
Path   = Feature_Function.getPath()
img    = Feature_Function.getImg(Path)
opt    = Feature_Detect.parse_opt(Path[0])
expN   = Feature_Detect.main(opt)

print("特徵擷取中...")
color    = Feature_Color.featureMainColor(Path,img)
stoolCnt = Feature_Segmentation.featureSegmentation(Path,expN)
blood    = Feature_Blood.bloodEvaluate(img)
bss      = Feature_BSS.bssEvaluate(img)
obj      = Feature_Object.featureObjDetect(Path,img,expN)
bubble   = Feature_BubbleFilter.featureBubbleDetect(Path,img,expN)
print("特徵擷取完畢!\n圖片:"+str(Path))
featureList = [color,stoolCnt,blood,bss,obj,bubble]
print(featureList)
healthPoint = Feature_Function.getHealthPoint(featureList)
healthResult = Feature_Function.getHealthResult(featureList,'\n')
print("HealthPoint: ",healthPoint,sep="")
print("HealthPoint: ",healthResult,sep="")
