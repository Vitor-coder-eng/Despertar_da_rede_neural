import cv2
import torch
import numpy as np
import pathlib
from pathlib import Path

# Corrige caminho para Windows
pathlib.PosixPath = pathlib.WindowsPath

# Carrega o modelo
path = 'best.pt'  # Certifique-se que está no mesmo diretório
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
model.conf = 0.6

# Captura da DroidCam - geralmente a câmera é 1 (mas pode ser 0 ou 2, teste se necessário)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Erro ao abrir a câmera. Verifique se o DroidCam está conectado.")
    exit()

print("Câmera conectada. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o frame.")
        break

    # Converte a imagem para RGB e detecta
    results = model(frame)
    frame_detected = np.squeeze(results.render())  # Renderiza as detecções

    cv2.imshow('Detecção com YOLO', frame_detected)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
