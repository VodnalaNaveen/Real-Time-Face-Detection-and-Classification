import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
import tensorflow
from tensorflow import keras
import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json


"""

for the default model path

python inference.py 

or 

python inference.py --image "path_to_image" --model "path_to_model" --scaleFactor "path to the scale factor" --minNeighbors "path to the min neighbors"



other model paths  

'mobilenet_scrath_model.keras',
 'mobilenet_trans1.keras',
 'mobilenet_trans2.keras',
 'resnet_scrath.keras',
 'resnet_trans1.keras',
 'resnet_trans2.keras',
 'vgg16_scratch.keras',
 'vgg16_transf1.keras',
 'vgg16_transf2.keras'


"""

with open('database/db.json', 'r') as file:
            db = json.load(file)
            file.close()
db =  {int(k): v for k, v in db.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='test data/frame_0_flip.jpg', help="Path to the image file")
    parser.add_argument("--model", type=str, default='models/resnet_trans1.keras', help="Path to the model file")
    parser.add_argument("--scaleFactor", type=float, default=1.2, help="Scale factor for the detection")
    parser.add_argument("--minNeighbors", type=int, default=5, help="Minimum neighbors for the detection")

    args = parser.parse_args()

    # Load the model
    model = keras.models.load_model(args.model)

    original_img = cv2.imread(args.image)
    if original_img is None:
        print("Error: image not found.")
        exit()

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces = classifier.detectMultiScale(gray, 
                                      scaleFactor=args.scaleFactor,
                                      minNeighbors=args.minNeighbors)

    if len(faces) == 0:
        print("No face detected.")
    else:
        # Process each detected face
        for (x, y, w, h) in faces:

            face_img = original_img[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            
            input_arr = keras.utils.img_to_array(face_img)
            input_arr = input_arr / 255.0
            input_arr = np.expand_dims(input_arr, axis=0)
            
            predictions = model.predict(input_arr)
            pred_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            predicted_class = db.get(pred_index)
            
            cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            label = f"{predicted_class} ({confidence:.2%})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(original_img, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
            # Add text
            cv2.putText(original_img, label, (x, y - 5), font, font_scale, (0, 0, 0), thickness)
            
            print(f"Detected face: {predicted_class} with confidence: {confidence:.2%}")

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


