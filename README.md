# Reconhecimento de Gatinhos
!pip install tensorflow opencv-python matplotlib
!pip install opencv-python torch torchvision
!git clone https://github.com/ultralytics/yolov5  # -> Clonar o repositório do YOLOv5
%cd yolov5
!pip install -r requirements.txt  # -> Instalar os requisitos do YOLOv5
!wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt  # -> Baixar o modelo pré-treinado
import torch
import cv2
from google.colab.patches import cv2_imshow # -> Para exibir a imagem no Colab

modelo = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # -> Carregar o modelo YOLOv5s
imagem_path = '/content/filhotes.jpg'  # -> Substitua pelo caminho da sua imagem
resultados = modelo(imagem_path)

# -> Filtrar os resultados para encontrar apenas gatos (classe 15 no COCO dataset)
gatos = resultados.xyxy[0][resultados.xyxy[0][:, -1] == 15]

# -> Exibir a imagem com as caixas delimitadoras dos gatos
imagem = cv2.imread(imagem_path)
for gato in gatos:
    x1, y1, x2, y2 = gato[:4].int().tolist() # -> Converte tensores para inteiros
    cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)  # -> Desenhar retângulo verde ao redor dos gatos

cv2_imshow(imagem)

# -> Contar o número de gatos
numero_gatos = len(gatos)
print("Número de gatos encontrados:", numero_gatos)
