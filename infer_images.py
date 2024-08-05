import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from model import SimpleCNN

def load_model(model_path):
    """
    Carrega o modelo treinado a partir do caminho especificado.

    Args:
        model_path (str): Caminho para o arquivo do modelo treinado.

    Returns:
        model: Modelo carregado em modo de avaliação.
    """
    model = SimpleCNN()  # Inicializa a arquitetura do modelo
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Carrega os pesos do modelo
    model.eval()  # Define o modelo para modo de avaliação
    return model

def preprocess_image(image_path, img_size=512):
    """
    Preprocessa a imagem de entrada: carrega, e transforma a imagem para o formato esperado pelo modelo.

    Args:
        image_path (str): Caminho para a imagem de entrada.
        img_size (int, optional): Tamanho para redimensionar a imagem. Padrão é 512.

    Returns:
        image: Imagem preprocessada como tensor.
    """
    image = Image.open(image_path).convert('RGB')  # Carrega a imagem e converte para RGB
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Redimensiona a imagem
        transforms.ToTensor()  # Converte a imagem para tensor
    ])
    image = transform(image).unsqueeze(0)  # Adiciona uma dimensão de batch
    return image

def postprocess_output(output):
    """
    Pós-processa a saída do modelo para uma imagem binarizada.

    Args:
        output (tensor): Saída do modelo.

    Returns:
        Image: Imagem binarizada.
    """
    output = output.squeeze().detach().numpy()  # Remove a dimensão de batch e converte para numpy
    output = (output > 0.5).astype(np.uint8) * 255  # Binariza a saída e escala para 0-255
    return Image.fromarray(output)  # Converte para imagem

def save_image(image, output_path):
    """
    Salva a imagem processada no caminho especificado.

    Args:
        image (Image): Imagem a ser salva.
        output_path (str): Caminho para salvar a imagem.
    """
    image.save(output_path)

def main(input_image_path, model_path, output_image_path, img_size=512):
    """
    Função principal que carrega o modelo, processa a imagem de entrada, faz a inferência e salva a imagem de saída.

    Args:
        input_image_path (str): Caminho para a imagem de entrada.
        model_path (str): Caminho para o modelo treinado.
        output_image_path (str): Caminho para salvar a imagem de saída.
        img_size (int, optional): Tamanho para redimensionar a imagem. Padrão é 512.
    """
    model = load_model(model_path)  # Carrega o modelo treinado
    image = preprocess_image(input_image_path, img_size)  # Preprocessa a imagem de entrada

    with torch.no_grad():  # Desabilita o cálculo de gradientes para inferência
        output = model(image)  # Faz a inferência com o modelo

    output_image = postprocess_output(output)  # Pós-processa a saída do modelo
    save_image(output_image, output_image_path)  # Salva a imagem de saída
    print(f"Inferência salva em: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz inferência em uma imagem usando um modelo treinado de segmentação de vegetação.")
    parser.add_argument('--input', type=str, required=True, help='Caminho para a imagem de entrada.')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo treinado.')
    parser.add_argument('--output', type=str, required=True, help='Caminho para salvar a imagem de saída.')

    args = parser.parse_args()  # Parsea os argumentos da linha de comando
    main(args.input, args.model, args.output)  # Executa a função principal