import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")

# Abre o vídeo para leitura
cap = cv2.VideoCapture("files/call_center.mp4")
# Se você quiser usar a câmera em tempo real, descomente a linha abaixo:
# cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

conf_threshold = 0.15
num_pessoas = 0

# Inicializa o gravador de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Formato MP4
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

while True:
    _, img = cap.read()
    results = model.predict(img, conf=conf_threshold)
    num_pessoas = 0
    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Coordenadas da caixa no formato (esquerda, superior, direita, inferior)
            c = box.cls
            class_name = model.names[int(c)]
            confidence = box.conf.item() 
            if model.names[int(c)] == 'person':
                annotator.box_label(b, f"({confidence:.2f})", (0, 0, 255))
                num_pessoas += 1
            img = annotator.result()

    h, w, _ = img.shape
    cv2.putText(img, f"Pessoas: {num_pessoas}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Adiciona a legenda na parte inferior direita
    cv2.putText(img, "POC CMC - FELIPE FEYH TEST", (w - 250, h - 20), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 2)

    # Grava o frame modificado no vídeo de saída
    out.write(img)

    cv2.imshow('YOLO V8 Detection', img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Libera os recursos
cap.release()
out.release()
cv2.destroyAllWindows()
