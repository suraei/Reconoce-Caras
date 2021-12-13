import cv2
import os

carpeta = "./Caras"
listaPersonas = os.listdir(carpeta)
print("Lista de personas: ", listaPersonas)

reconocimiento = cv2.face.EigenFaceRecognizer_create()

#Leer modelo
reconocimiento.read("modelocaras.xml")

webcam = cv2.VideoCapture(0)

archivo_cara = "cara.xml"
clasificador_cara = cv2.CascadeClassifier(archivo_cara)

while True:
    (_,cam) = webcam.read()
    if cam is None:
        print("No se pudo acceder a la webcam")
        break
    gris = cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
    auxCam = gris.copy()

    caras = clasificador_cara.detectMultiScale(gris,1.5,5)

    for(x,y,w,h) in caras:
        rostro = auxCam[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        resultado = reconocimiento.predict(rostro)

        cv2.putText(cam,'{}'.format(listaPersonas[resultado[0]]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

    cv2.imshow("Camara",cam)
    q = cv2.waitKey(1)
    if q == 27:
        break