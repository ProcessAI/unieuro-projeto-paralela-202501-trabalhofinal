import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# Função para carregar e pré-processar a imagem
def preprocess_image(image_path):
    # Carrega a imagem
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplica um filtro Gaussiano para remover ruído
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Binariza usando limiar adaptativo
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    return binary

# Função para contar estrelas
def count_stars(binary_img):
    # Inverte a imagem (estrelas devem estar em branco)
    binary_img = cv2.bitwise_not(binary_img)
    
    # Usa rotulagem de componentes conectados
    labels = measure.label(binary_img, connectivity=2)
    
    # Conta quantos objetos foram encontrados
    star_count = len(np.unique(labels)) - 1  # subtrai 1 para excluir o fundo (label 0)
    
    return star_count, labels

# Caminho da imagem (substitua pelo caminho da sua imagem)
image_path = './exemplo.jpg'

# Pré-processamento
binary = preprocess_image(image_path)

# Contagem
num_stars, labeled_image = count_stars(binary)

# Resultado
print(f"Número de estrelas detectadas: {num_stars}")

# (Opcional) Visualização
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagem binarizada")
plt.imshow(binary, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Estrelas rotuladas")
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.show()
