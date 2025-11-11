import face_recognition
import cv2
import os
import numpy as np

# Cargar imágenes de referencia
known_encodings = []
known_names = []

for file in os.listdir("known"):
    img = face_recognition.load_image_file(f"known/{file}")
    encoding = face_recognition.face_encodings(img)[0]
    known_encodings.append(encoding)
    known_names.append(os.path.splitext(file)[0])

# Cámara
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for face_encoding, (top, right, bottom, left) in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Desconocido"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
