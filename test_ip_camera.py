# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:34:19 2024

@author: U170021
"""

import cv2

#"http://skycam.sebewainggigvillage.com/mjpg/video.mjpg"

url_camera = "http://skycam.sebewainggigvillage.com/mjpg/video.mjpg"  # Substitua pelo URL da sua câmera
cap = cv2.VideoCapture(url_camera)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Câmera IP", frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
