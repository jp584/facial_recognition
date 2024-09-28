import cv2
import numpy as np
from PIL import Image
import os
# ML model
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Path for out data set
path = "datasets"

# Method to train the opencv model
def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        faceNP = np.array(faceImage)
        Id= (os.path.split(imagePaths)[-1].split(".")[1])
        Id=int(Id)
        faces.append(faceNP)
        ids.append(Id)
        cv2.imshow("Entrenando",faceNP)
        cv2.waitKey(1)
    return ids, faces

IDs, facedata = getImageID(path)
recognizer.train(facedata, np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("¡Entrenamiento completado!")