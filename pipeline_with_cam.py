from pipeline.pipeline import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--objdet_path', type=str, default="haarcascade_frontalface_default.xml", help='Path to the object detection model')
    parser.add_argument('--imgcls_path', type=str, default="models/vgg16_transf1.keras", help='Path to the image classification model')
    parser.add_argument('--source_path', type=int, default=0, help='Path to the video source')

    args = parser.parse_args()

    pipeline1 = pipeline(objdet_path=args.objdet_path,
                         imgcls_path=args.imgcls_path,
                         source_path=args.source_path)
    objdetmodel = pipeline1.load_objectdetection_model()
    imgclsmodel = pipeline1.load_imageclassification_model()
    database = pipeline1.load_database()
    pipeline1.run(model1=objdetmodel,model2=imgclsmodel,database=database)




