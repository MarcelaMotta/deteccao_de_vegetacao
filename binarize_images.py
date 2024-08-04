import argparse
import os
from PIL import Image
import numpy as np

def binarize_image(image_path, output_path):
    """
    Binariza uma imagem usando o índice ExG e salva como imagem em escala de cinza.

    Args:
        image_path (str): Caminho para a imagem RGB.
        output_path (str): Caminho para salvar a imagem binarizada.
    """
    # Abrir a imagem
    image = Image.open(image_path)
    image = image.convert("RGB")  # Garantir que a imagem esteja no modo RGB
    np_image = np.array(image)

    # Calcular o índice ExG
    R = np_image[:, :, 0].astype(np.float32)
    G = np_image[:, :, 1].astype(np.float32)
    B = np_image[:, :, 2].astype(np.float32)
    ExG = 2 * G - R - B

    # Binarizar a imagem
    binary_image = np.where(ExG > 0, 1, 0).astype(np.uint8) * 255

    # Converter para imagem PIL e salvar
    binarized_image = Image.fromarray(binary_image, mode='L')
    binarized_image.save(output_path)

    print(f'Imagem binarizada salva em: {output_path}')

def binarize_images(input_folder, output_folder):
    """
    Binariza todas as imagens em um diretório e salva as imagens binarizadas em um diretório de saída.

    Args:
        input_folder (str): Diretório que contém as imagens RGB em blocos.
        output_folder (str): Diretório onde serão salvas as imagens segmentadas em escala cinza.
    """
    # Criar a pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)

    # Iterar sobre todas as imagens no diretório de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            binarize_image(input_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Binariza imagens RGB em blocos e salva como imagens em escala de cinza.")
    parser.add_argument('--input', type=str, required=True, help='Caminho para o diretório que contém as imagens RGB em blocos.')
    parser.add_argument('--output', type=str, required=True, help='Caminho para o diretório onde serão salvas as imagens segmentadas em escala cinza.')

    args = parser.parse_args()

    binarize_images(args.input, args.output)

if __name__ == '__main__':
    main()