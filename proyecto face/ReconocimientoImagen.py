import cv2
import os

dataPath = 'C:/Users/USUARIO PC/Documents/proyecto face/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

# Cambia la ruta de la imagen que deseas utilizar
image_path = 'C:/Users/USUARIO PC/Documents/GitHub/ProyectoFace/proyecto face/Grupal.jpg'
frame = cv2.imread(image_path)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
auxFrame = gray.copy()

faces = faceClassif.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    rostro = auxFrame[y:y+h, x:x+w]
    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
    result = face_recognizer.predict(rostro)

    cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

	
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
