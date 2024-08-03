import argparse
from PIL import Image
import os

def divide_image(image_path, output_folder, block_size):
    """
    Divide uma imagem em blocos menores e salva cada bloco como um arquivo PNG.

    Args:
        image_path (str): Caminho para a imagem TIFF.
        output_folder (str): Pasta onde os blocos serão salvos.
        block_size (int): Tamanho dos blocos (em pixels).
    """
    # Abrir a imagem
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Criar a pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)

    # Iterar sobre a imagem e salvar blocos
    for i in range(0, img_width, block_size):
        for j in range(0, img_height, block_size):
            # Calcular as coordenadas da janela
            box = (i, j, i + block_size, j + block_size)
            block = image.crop(box)

            # Nome do arquivo de saída
            block_filename = os.path.join(output_folder, f'block_{i}_{j}.png')

            # Salvar o bloco como PNG
            block.save(block_filename)

    print(f'Imagem dividida em blocos de {block_size}x{block_size} e salva em {output_folder}.')

def main():
    parser = argparse.ArgumentParser(description="Divide uma imagem TIFF em blocos menores e salva como arquivos PNG.")
    parser.add_argument('--input', type=str, required=True, help='Caminho para a imagem TIFF.')
    parser.add_argument('--output', type=str, required=True, help='Pasta onde os blocos serão salvos.')
    parser.add_argument('--block-size', type=int, default=512, help='Tamanho dos blocos em pixels.')

    args = parser.parse_args()

    divide_image(args.input, args.output, args.block_size)

if __name__ == '__main__':
    main()