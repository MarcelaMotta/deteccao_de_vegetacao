import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from dataset import VegetationDataset
from model import SimpleCNN

def set_seed(seed):
    """
    Define a seed para garantir a reprodutibilidade.

    Args:
        seed (int): Valor da seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(images_dir, labels_dir, model_path, img_size=512, batch_size=2, epochs=20, learning_rate=0.001, seed=42):
    """
    Treina o modelo de segmentação de vegetação usando uma rede neural convolucional simples.

    Args:
        images_dir (str): Caminho para o diretório que contém as imagens RGB.
        labels_dir (str): Caminho para o diretório que contém as imagens segmentadas.
        model_path (str): Caminho para salvar o modelo treinado.
        img_size (int, optional): Tamanho para redimensionar as imagens. Padrão é 512.
        batch_size (int, optional): Tamanho do batch para o DataLoader. Padrão é 2.
        epochs (int, optional): Número de épocas para treinar. Padrão é 20.
        learning_rate (float, optional): Taxa de aprendizado para o otimizador. Padrão é 0.001.
        seed (int, optional): Seed para reprodutibilidade. Padrão é 42.
    """
    set_seed(seed)  # Define a seed para reprodutibilidade

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Redimensiona as imagens para img_size x img_size
        transforms.ToTensor()  # Converte as imagens para tensores
    ])

    dataset = VegetationDataset(images_dir, labels_dir, transform)

    # Divide o dataset em treino (80%), validação (10%) e teste (10%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()  # Função de perda binária
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Otimizador Adam

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()  # Define o modelo para modo de treino
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zera os gradientes do otimizador
            outputs = model(images)  # Passa as imagens pelo modelo
            loss = criterion(outputs, labels)  # Calcula a perda
            loss.backward()  # Propaga o erro para ajustar os pesos
            optimizer.step()  # Atualiza os pesos do modelo
            running_loss += loss.item() * images.size(0)  # Acumula a perda

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validação
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)
        all_outputs = (all_outputs > 0.5).astype(int)  # Binariza as saídas

        precision = precision_score(all_labels.flatten(), all_outputs.flatten())
        recall = recall_score(all_labels.flatten(), all_outputs.flatten())
        f1 = f1_score(all_labels.flatten(), all_outputs.flatten())

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        # Salva o modelo com o número da época no nome do arquivo
        epoch_model_path = f"{model_path}_epoch_{epoch+1}.h5"
        torch.save(model.state_dict(), epoch_model_path)
        print(f'Modelo salvo em: {epoch_model_path}')

    # Teste
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    all_outputs = (all_outputs > 0.5).astype(int)

    precision = precision_score(all_labels.flatten(), all_outputs.flatten())
    recall = recall_score(all_labels.flatten(), all_outputs.flatten())
    f1 = f1_score(all_labels.flatten(), all_outputs.flatten())

    print(f'Test Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

def main():
    """
    Função principal para parsear argumentos da linha de comando e iniciar o treinamento.
    """
    parser = argparse.ArgumentParser(description="Treina um modelo de segmentação de vegetação usando uma rede neural convolucional simples.")
    parser.add_argument('--rgb', type=str, required=True, help='Caminho para o diretório que contém as imagens RGB em blocos.')
    parser.add_argument('--groundtruth', type=str, required=True, help='Caminho para o diretório que contém as imagens segmentadas.')
    parser.add_argument('--modelpath', type=str, required=True, help='Caminho para salvar o modelo treinado.')

    args = parser.parse_args()

    train_model(args.rgb, args.groundtruth, args.modelpath)

if __name__ == '__main__':
    main()