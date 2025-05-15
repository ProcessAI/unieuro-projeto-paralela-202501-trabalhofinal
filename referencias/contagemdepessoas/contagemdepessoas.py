import torch
import cv2
import numpy as np

# Carregar o modelo YOLOv5 pré-treinado (pode ser um modelo para detecção geral)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' é uma versão leve

# Carregar a imagem aérea
image_path = 'pessoas2.jpg'  # Substitua pelo caminho da sua imagem
img = cv2.imread(image_path)

# Fazer a detecção de objetos na imagem
results = model(img)

# Mostrar os resultados (bboxes, confidência, classes detectadas)
results.show()

# Obter as deteções
detections = results.pandas().xywh[0]  # Pandas DataFrame com informações sobre as detecções
people_count = 0

# Filtrando as detecções para contar apenas as pessoas (classe 0 é a classe 'person' no COCO dataset)
for index, row in detections.iterrows():
    if row['name'] == 'person':
        people_count += 1

print(f'Número de pessoas detectadas: {people_count}')
