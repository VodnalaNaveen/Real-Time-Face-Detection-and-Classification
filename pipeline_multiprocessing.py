import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  

import cv2
import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json


class Pipeline:

    def __init__(self, objdet_path, imgcls_path, source_path):
        self.objdet_path = objdet_path  
        self.imgcls_path = imgcls_path  
        self.source_path = source_path  

    
    def load_objectdetection_model(self):
        detector = cv2.CascadeClassifier(self.objdet_path)
        print("Object detection model loaded successfully.")
        return detector

    
    def load_imageclassification_model(self):
        model = keras.models.load_model(self.imgcls_path)
        print("Image classification model loaded successfully.")
        return model

    def load_database(self):
        with open('database/db.json', 'r') as file:
            db = json.load(file)
            file.close()
        db =  {int(k): v for k, v in db.items()}
        return db
    

    def get_window_details(self):
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window", 1920, 1080)
        screen_size = cv2.getWindowImageRect("window")
        cv2.destroyWindow("window")
        return screen_size[2], screen_size[3] 

    def display_single_video(self, video_path, window_name, objdet_path, imgcls_path, window_width, window_height, db):
        
        detector = cv2.CascadeClassifier(objdet_path)
        imgclassmodel = keras.models.load_model(imgcls_path)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_FPS, 5)
        while True:
            ret, frame = video.read()
            if not ret:
                break

            resize_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
            gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
            results  = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(results) >= 1:
                for index in range(len(results)):
                    x, y, w, h = results[index][0]*2, results[index][1]*2, results[index][2]*2, results[index][3]*2
                    cropped_face = frame[y:y+h, x:x+w]

                    if cropped_face.size != 0 and cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                        resized = cv2.resize(cropped_face, (224, 224))
                        arr = np.array([resized]) 
                        prediction = imgclassmodel.predict(arr)
                        class_idx = np.argmax(prediction, axis=1)
                        class_name = db.get(class_idx[0])
                        cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        video.release()
        
    def display_two_videos(self):
        db = self.load_database()
        window_width, window_height = self.get_window_details()

        screen_width = window_width // 2
        screen_height = window_height

        multiprocessing.set_start_method("spawn", force=True)

        process1 = multiprocessing.Process(
            target=self.display_single_video,
            args=(self.source_path[0], "video1", self.objdet_path, self.imgcls_path, screen_width, screen_height, db)
        )
        process2 = multiprocessing.Process(
            target=self.display_single_video,
            args=(self.source_path[1], "video2", self.objdet_path, self.imgcls_path, screen_width, screen_height, db)
        )

        process1.start()
        process2.start()
        process1.join()
        process2.join()


# --- Run the pipeline ---q
if __name__ == "__main__":
    objdet_model_path = "haarcascade_frontalface_default.xml"
    imgcls_model_path = "models/vgg16_transf1.keras" 
    video_paths = [0,1]  

    pipeline = Pipeline(objdet_path = objdet_model_path, imgcls_path = imgcls_model_path, source_path = video_paths)
    pipeline.display_two_videos()
