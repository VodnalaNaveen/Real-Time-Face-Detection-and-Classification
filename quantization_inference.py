import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow
import cv2
import json
from keras.applications.mobilenet import preprocess_input  # or vgg16, if trained that way

# Load TFLite model
interpreter = tensorflow.lite.Interpreter(model_path="quantization_models/mobilenet_trans1_fp16.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()
output_index = interpreter.get_output_details()

class pipeline:

    def __init__(self, objdet_path, source_path):
        self.objdet_path = objdet_path  # path to object detection algorithm
        self.source_path = source_path  # path to the source (video)

    # loading object detection model
    def load_objectdetection_model(self):
        detector = cv2.CascadeClassifier(self.objdet_path)
        print("object detection model has been loaded successfully")
        return detector

    # loading the database
    def load_database(self):
        with open("database/db.json", "r") as file:
            db = json.load(file)
        db = {int(key): value for key, value in db.items()}
        return db

    # process the video
    def process_video(self, db):
        video = cv2.VideoCapture(self.source_path)
        video.set(cv2.CAP_PROP_FPS, 15)
        while True:
            ret, frame = video.read()
            if ret:
                print("Processing frame...")
                resize_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                print("Resized frame to half size.")
                gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)

                print("Converted frame to grayscale.")
                results = self.load_objectdetection_model().detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in results:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 4)
                print("Detected faces:", results)

                if len(results) >= 1:
                    x, y, w, h = results[0][0]*2, results[0][1]*2, results[0][2]*2, results[0][3]*2
                    h_frame, w_frame, _ = frame.shape
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(x + w, w_frame), min(y + h, h_frame)
                    if y2 > y and x2 > x:
                        cropped_face = frame[y:y2, x:x2]
                        try:
                            resized = cv2.resize(cropped_face, (224, 224))
                            resized = np.expand_dims(resized, axis=0).astype(np.float32)


                            print("Processing face crop at coordinates:", x, y, x2, y2)

                            interpreter.invoke()
                            interpreter.set_tensor(input_index[0]['index'], resized)

                            predictions = interpreter.get_tensor(output_index[0]['index'])
                            class_idx = np.argmax(predictions)
                            class_name = db.get(class_idx[0])

                            cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 4)
                        except Exception as e:
                            print("Error in face processing:", e)
                    else:
                        print("Invalid face crop coordinates:", x, y, x2, y2)

                cv2.imshow("window", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cv2.destroyAllWindows()
        video.release()

    # run
    def run(self, database):
        self.process_video(database)

# ===============================
# Example to run the pipeline
# ===============================

pipeline1 = pipeline(
    objdet_path=r"C:\Users\prema\Downloads\project - Copy\haarcascade_frontalface_default.xml",
    source_path=1
)
database = pipeline1.load_database()
model1 = pipeline1.load_objectdetection_model()
pipeline1.run( database)
