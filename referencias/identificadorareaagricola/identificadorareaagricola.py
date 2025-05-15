import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carregar a imagem (substitua pelo caminho da sua imagem de satélite)
image = cv2.imread('exemplo1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionar para facilitar o processamento
image_resized = cv2.resize(image_rgb, (400, 400))

# Converter a imagem para um vetor 2D de pixels (cada pixel é uma linha com suas 3 cores RGB)
pixels = image_resized.reshape((-1, 3))

# Aplicando o K-means para segmentação
kmeans = KMeans(n_clusters=3, random_state=42)  # Vamos criar 3 clusters (uma para agricultura, outra para floresta e outra para fundo)
kmeans.fit(pixels)

# Prever os clusters para cada pixel
segmented_image = kmeans.predict(pixels)

# Reformatar a imagem segmentada para o formato original (resolução 400x400x3)
segmented_image = segmented_image.reshape(image_resized.shape[:2])

# Visualizar a imagem segmentada
plt.imshow(segmented_image, cmap='viridis')
plt.title('Imagem Segmentada - K-means')
plt.colorbar()
plt.show()

# Identificando as áreas agrícolas (supondo que o cluster de índice 1 é agricultura)
# Vamos marcar a área de agricultura em branco e o resto em preto para visualização
agriculture_cluster = 1  # Índice do cluster que corresponde à área agrícola
agriculture_mask = (segmented_image == agriculture_cluster)

# Mostrar a máscara agrícola
plt.imshow(agriculture_mask, cmap='gray')
plt.title('Área Agrícola Identificada')
plt.show()
