import cv2

archivo_cara = 'cara.xml'

clasificador_cara = cv2.CascadeClassifier(archivo_cara)
webcam = cv2.VideoCapture(0)

while True:
    (_, cam) = webcam.read()
    grises = cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
    caras = clasificador_cara.detectMultiScale(grises)
    
    for(x,y,w,h) in caras:
        cv2.rectangle(cam, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("OpenCV", cam)

    q = cv2.waitKey(10)

    if(q == 27):
        break