import cv2
import os
import imutils

nombrePersona = input("Ingrese el nombre de la persona: ")
carpeta = './Caras'
carpeta_persona = carpeta + '/' + nombrePersona

if not os.path.exists(carpeta_persona):
    print("Creando carpeta: " + carpeta_persona)
    os.makedirs(carpeta_persona)

webcam = cv2.VideoCapture(0)

archivo_cara = 'cara.xml'

clasificador_cara = cv2.CascadeClassifier(archivo_cara)
contador = 0

while True:
    (_, cam) = webcam.read()
    if cam is None:
        break 
    cam = imutils.resize(cam, width=640)
    grises = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    auxCam = cam.copy()

    caras = clasificador_cara.detectMultiScale(grises, 1.3, 5)
    for (x, y, w, h) in caras:
        cv2.rectangle(cam, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxCam[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(carpeta_persona + '/' + str(contador) + '.jpg', rostro)
        contador += 1

    cv2.imshow("Camara", cam)

    q = cv2.waitKey(1)
    if q == 27:
        break