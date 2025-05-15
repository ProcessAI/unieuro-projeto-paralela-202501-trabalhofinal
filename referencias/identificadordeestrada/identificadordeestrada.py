import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem de satélite
img = cv2.imread('free-sat-imgs.jpg.webp')

# Converter a imagem para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar o filtro de Canny para detectar bordas (estradas geralmente têm bordas bem definidas)
edges = cv2.Canny(gray, 100, 200)

# Usar um filtro de dilatação para tornar as bordas mais visíveis
dilated = cv2.dilate(edges, None, iterations=1)

# Exibir a imagem original e a imagem com as bordas detectadas
plt.figure(figsize=(10, 5))

# Imagem original
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis('off')

# Imagem com bordas detectadas
plt.subplot(1, 2, 2)
plt.imshow(dilated, cmap='gray')
plt.title("Estradas Detectadas")
plt.axis('off')

plt.show()
