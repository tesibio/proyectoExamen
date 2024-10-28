import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("no se pudo capturar el video")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    #Calcular FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    #Mostrar los FPS en pantalla
    #print(f'FPS: {fps}')
    cv2.putText(img,f'FPS:{int(fps)}',(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    #Mostrar la imagen con los FPS
    cv2.imshow("Deteccion de manos",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
