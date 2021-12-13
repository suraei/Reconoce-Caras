import cv2
import os
import numpy as np

carpeta = "./Caras"
listaPersonas = os.listdir(carpeta)
print("Lista de personas: ", listaPersonas)

etiquetas = []
datosCaras = []
etiqueta = 0

for nombreRostro in listaPersonas:
    Persona = carpeta + "/" + nombreRostro
    print("Leyendo imágenes...")

    for archivo in os.listdir(Persona):
        print("Rostros: " , nombreRostro + "/" + archivo)
        etiquetas.append(etiqueta)
        datosCaras.append(cv2.imread(Persona + "/" + archivo, 0))
        imagen = cv2.imread(Persona + "/" + archivo, 0)
       # cv2.imshow("Rostro", imagen)
       # cv2.waitKey(1)

    etiqueta += 1
#cv2.destroyAllWindows()

reconocimiento = cv2.face.EigenFaceRecognizer_create()

print("Entrenando...")
reconocimiento.train(datosCaras, np.array(etiquetas))

#Almacenamiento de la cara reconocida
reconocimiento.write("modelocaras.xml")
print("Modelo almacenado")