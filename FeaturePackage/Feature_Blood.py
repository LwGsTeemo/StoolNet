'''訓練照片中是否含有血液,並將其訓練結果儲存下來
'''
import numpy as np  
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D
import glob,os

def modelTrain():
    Path0 = glob.glob(os.getcwd()+'\data\Blood\\0\*.jpg')
    Path1 = glob.glob(os.getcwd()+'\data\Blood\\1\*.jpg')
    X_train=[]
    Y_train=[]
    for i in Path0:
        image = cv2.imread(i)
        img = img_to_array(image)
        X_train.append(img)
        Y_train.append(0)
    for i in Path1:
        image = cv2.imread(i)
        img = img_to_array(image)
        X_train.append(img)
        Y_train.append(1)

    X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.2, 
    random_state=0)

    y_TrainOneHot = np_utils.to_categorical(Y_train) 
    y_TestOneHot = np_utils.to_categorical(Y_test) 

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train_2D = X_train.astype('float32')  
    X_test_2D = X_test.astype('float32')  

    x_Train_norm = X_train_2D/255
    x_Test_norm = X_test_2D/255
    
    # 建立簡單的線性執行的模型
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(500,500,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=256, kernel_initializer='normal', activation='relu')) 
    model.add(Dropout(0.25))
    model.add(Dense(units=64, kernel_initializer='normal', activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(units=16, kernel_initializer='normal', activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(units=2, kernel_initializer='normal', activation='sigmoid'))
    model.add(Flatten())
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=50, batch_size=128, verbose=1) 
    model.save('Blood.h5')


    # 顯示訓練成果(分數)
    model = tf.keras.models.load_model('Blood.h5')
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    scores = model.evaluate(x_Test_norm, y_TestOneHot)  
    print("\n[B0843020] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

    # 預測(prediction)
    X = x_Test_norm[0:10,:]
    predictions = np.argmax(model.predict(X), axis=-1)
    print(predictions)

def bloodEvaluate(img):
    X_test=img.reshape(1,500,500,3)
    X_test = np.array(X_test)
    X_test_2D = X_test.astype('float32') 
    x_Test_norm = X_test_2D/255

    model = tf.keras.models.load_model('Blood.h5')
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    # scores = model.evaluate(x_Test_norm, y_TestOneHot)  
    # print("\n[B0843020] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  
    # 預測(prediction)
    predictions=np.argmax(model.predict(x_Test_norm),axis=-1)
    return int(predictions)
