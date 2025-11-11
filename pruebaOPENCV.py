import cv2 
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cam = cv2.VideoCapture(0)


while True:
    ret, frame = cam.read()

    if not ret:
        break


        # Convertir BGR (OpenCV) a RGB (MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hands = mp_hands.Hands() 
    result = hands.process(rgb)

    # Dibujar landmarks si detecta manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe Hands", frame)


    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()