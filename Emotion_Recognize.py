import cv2,numpy as np
from tensorflow.keras import models
from tensorflow.keras import models,layers
from tensorflow.keras.optimizers import Adam,SGD
import os

name=os.listdir("./Models")[-1]

def get_Model():
    Model=models.Sequential([
    layers.Conv2D(16,(2,2),input_shape=(48,48,3),activation='relu'),
    layers.Conv2D(32,(2,2),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,(2,2),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(128,(2,2),activation='relu'),
    # layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(36,activation='relu'),
    layers.Dense(6,activation='softmax')
    ])

    # Model.compile(Adam(learning_rate=0.1),loss='binary_crossentropy',metrics=['accuracy'])
    Model.compile(Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    return Model


def ISFace(img):
    face = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
    gre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face123 = face.detectMultiScale(gre,1.1,4)
    # print(face123)
    return True if len(face123)>0 else False

path="./Models/"+name
M=get_Model()
M.load_weights(path)
Class=['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
im=[]

cam=cv2.VideoCapture(0)
while True:
    _,img=cam.read()
    if ISFace(img):
        face = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
        gre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face123 = face.detectMultiScale(gre,1.1,4)
        # vid = cv2.VideoCapture(0)
        for(x,y,w,h) in face123 :
            cimage=img[y:y+h,x:x+w]
            simage=img[y:y+h,x:x+w]
            simage=cv2.resize(simage,(500,500))
            cimage=cv2.resize(cimage,(48,48))
            i=np.array(cimage)/255
            i=i.reshape((-1,48,48,3))
            cimage=i
            pr=M.predict([cimage])
            i=np.argmax(pr)
            cas=Class[i]
            pr=int(pr[0][i]*100)
            if pr>80:
                cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
            cv2.putText(img,cas+" "+str(pr)+"%",(x+w,y+h),5,1,(255,255,255),2)
            # print(tf.estimator.evaluate())
        cv2.imshow("CImage",simage)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    



