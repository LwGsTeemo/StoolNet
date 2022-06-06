'''利用Alexnet的內容去做糞便的分類訓練,依照''布里斯托糞便分類法''將其(依其軟硬度及形狀)分類成7類
'''
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

def Pathfile():
    dataPath = os.path.join(os.path.dirname(__file__), 'data/images')
    modelPath = os.path.join(os.path.dirname(__file__), 'BSS.h5')
    accPicPath = os.path.join(os.path.dirname(__file__),'BSS_acc.png')
    lossPicPath = os.path.join(os.path.dirname(__file__),'BSS_loss.png')
    return dataPath,modelPath,accPicPath,lossPicPath

def load_pic(path):
    dataPath,modelPath,accPicPath,lossPicPath = Pathfile()
    x, y = list(), list()
    for dirname,dirnames,filenames in os.walk(dataPath):
        if(not dirnames):
            for i in filenames:
                imgpath = os.path.join(dirname, i)
                bgr = cv2.imread(imgpath)
                rgb = bgr[:,:,::-1]
                rgb = cv2.resize(rgb, (500,500), interpolation=cv2.INTER_AREA)
                x.append(rgb)
                y.append(int(dirname.split(sep='\\')[-1])-1)
    temp = list(zip(x, y)) 
    temp_train, temp_test = train_test_split(temp, random_state=19, train_size=0.7)
    x_train, y_train = zip(*temp_train)
    x_test, y_test = zip(*temp_test)
    x_train = np.asarray(x_train, dtype='float32') / 255
    y_train = np_utils.to_categorical(np.asarray(y_train))
    x_test = np.asarray(x_test, dtype='float32') / 255
    y_test = np_utils.to_categorical(np.asarray(y_test))
    return x_train, y_train, x_test, y_test

def BssModelTrain():
    dataPath,modelPath,accPicPath,lossPicPath = Pathfile()
    x_train, y_train, x_test, y_test = load_pic(dataPath)
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same', input_shape=(500, 500, 3), activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(597, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(597, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    history = model.fit(x=x_train, y=y_train, validation_data=[x_test, y_test], epochs=150, batch_size=10, verbose=1)
    model.save(modelPath)
    plt.figure(figsize=(30,18))
    plt.plot(history.epoch, history.history['loss'])
    plt.plot(history.epoch, history.history['val_loss'])
    plt.title('loss')
    plt.savefig(lossPicPath)
    plt.figure(figsize=(30,18))
    plt.plot(history.epoch, history.history['acc'])
    plt.plot(history.epoch, history.history['val_acc'])
    plt.title('accuracy')
    plt.savefig(accPicPath)

def bssEvaluate(img):
    X_test=img.reshape(1,500,500,3)
    X_test = np.array(X_test)
    X_test_2D = X_test.astype('float32') 
    x_Test_norm = X_test_2D/255

    model = keras.models.load_model('BSS_V2.h5')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # scores = model.evaluate(x_Test_norm, y_TestOneHot)  
    # print("\n[B0843020] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  
    # 預測(prediction)
    # predictions=np.argmax(model.predict(x_Test_norm),axis=-1)
    predictions=np.argmax(model.predict(x_Test_norm),axis=-1)+1
    return int(predictions)