import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow
from tensorflow import keras
import cv2
import json

class pipeline:

    def __init__(self,objdet_path,imgcls_path,source_path):

        self.objdet_path = objdet_path # path to object detection algorithm
        self.imgcls_path = imgcls_path # path to image classification algorithm
        self.source_path = source_path # path to the source (video)


    # loading object detection model
    def load_objectdetection_model(self):
        detector = cv2.CascadeClassifier(self.objdet_path)
        print("object detection model has been loaded successfully")
        return detector
    
    # loading image classificaiton model
    def load_imageclassification_model(self):
        model = keras.models.load_model(self.imgcls_path)
        print("image classification model has been loaded successfully")
        return model
    
    # loading the database
    def load_database(self):
        with open("database/db.json","r") as file:
            db = json.load(file)
            file.close()
        db = {int(key):value for key,value in db.items()}
        return db
    
    # process the video
    def process_video(self,objectdetmodel,imgclassmodel,db):
        video = cv2.VideoCapture(self.source_path)
        video.set(cv2.CAP_PROP_FPS, 15)
        while True:
            ret,frame = video.read()
            if ret:
                resize_frame = cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
                gray = cv2.cvtColor(resize_frame,cv2.COLOR_BGR2GRAY)
                results = objectdetmodel.detectMultiScale(gray,1.3,5) # loading the frame to object detecton algorithm
                if len(results) >=1:
                    x,y,w,h = results[0][0]*2,results[0][1]*2,results[0][2]*2,results[0][3]*2
                    cropped_face = resize_frame[y:y+h,x:x+w]
                    resized = cv2.resize(cropped_face, (224, 224)) 
                    arr = np.array([resized]) # adding batch size
                    results = imgclassmodel.predict(arr) # make prediction
                    class_idx = np.argmax(results,axis=1)
                    class_name = db.get(class_idx[0])
                    cv2.putText(frame,class_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
                cv2.imshow("window",frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cv2.destroyAllWindows()
        video.release()

    # run
    def run(self,model1,model2,database):
        self.process_video(model1,model2,database)