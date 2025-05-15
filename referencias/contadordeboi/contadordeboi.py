import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função de clique para capturar valores HSV dos pixels clicados
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = param[y, x]
        print(f"HSV em ({x}, {y}): {pixel}")

# Função para processar e segmentar a imagem
def processar_imagem(imagem_path):
    imagem = cv2.imread(imagem_path)
    
    # Verificar se a imagem foi carregada corretamente
    if imagem is None:
        print(f"Erro ao carregar a imagem em: {imagem_path}")
        return None, None
    
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    return imagem, imagem_hsv

def segmentar_bois(imagem, imagem_hsv, cor_boi_inferior, cor_boi_superior):
    # Criar uma máscara para isolar as áreas da imagem que correspondem à cor branca
    mascara_bois = cv2.inRange(imagem_hsv, cor_boi_inferior, cor_boi_superior)
    
    # Aplicar a máscara à imagem original para destacar os bois
    imagem_segmentada = cv2.bitwise_and(imagem, imagem, mask=mascara_bois)
    
    return mascara_bois, imagem_segmentada

def detectar_contornos(mascara_bois):
    kernel = np.ones((5, 5), np.uint8)
    mascara_bois = cv2.morphologyEx(mascara_bois, cv2.MORPH_OPEN, kernel)
    mascara_bois = cv2.morphologyEx(mascara_bois, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(mascara_bois, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def desenhar_contornos(imagem, contornos):
    imagem_contornada = imagem.copy()
    bois_detectados = 0
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if 1000 < area < 10000:  # Ajuste conforme necessário
            bois_detectados += 1
            cv2.drawContours(imagem_contornada, [contorno], -1, (0, 255, 0), 2)
    return imagem_contornada, bois_detectados

# Função para contar bois
def contar_bois(imagem_path):
    imagem, imagem_hsv = processar_imagem(imagem_path)
    
    # Verificar se a imagem foi carregada corretamente
    if imagem is None or imagem_hsv is None:
        return

    # Definir intervalo inicial para o branco (pode ser ajustado)
    cor_boi_inferior = np.array([0, 0, 220])
    cor_boi_superior = np.array([180, 20, 255])

    # Segmentação e contagem dos bois
    mascara_bois, imagem_segmentada = segmentar_bois(imagem, imagem_hsv, cor_boi_inferior, cor_boi_superior)
    contornos = detectar_contornos(mascara_bois)
    imagem_contornada, bois_detectados = desenhar_contornos(imagem, contornos)

    # Exibir a imagem original e a máscara binária
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Máscara Binária (HSV)")
    plt.imshow(mascara_bois, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Exibir a imagem segmentada (onde os bois são detectados)
    imagem_segmentada_rgb = cv2.cvtColor(imagem_segmentada, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_segmentada_rgb)
    plt.title("Imagem Segmentada")
    plt.axis("off")
    plt.show()

    # Exibir o número de bois detectados
    print(f"Bois detectados: {bois_detectados}")

# Caminho da sua imagem
imagem_path = './bois.jpg'

# Se você quiser apenas explorar os valores HSV, use a função explorar_imagem
# explorar_imagem(imagem_path)

# Contar bois na imagem (você pode rodar isso depois de ajustar a máscara)
contar_bois(imagem_path)
