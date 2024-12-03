# Standard Python libraries
import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Count the total number of weights in the model
def count_weights(model):
    total_weights = sum(param.numel() for param in model.parameters() if param.requires_grad and param.dim() > 1)
    return total_weights

def print_quantized_weights(model):
    weights_list = []  # 用于存储所有量化层的 int8 权重

    for name, module in model.named_modules():

        if isinstance(module, (torch.nn.quantized.Conv1d, torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
            print(f"Extracting weights from layer: {name}")

            weight_int_repr = module.weight().int_repr().cpu().numpy().flatten()  # 将权重提取并展开为一维数组
            weights_list.append(weight_int_repr)  # 将 int8 权重数组添加到列表中

            print(f"Weight Shape: {weight_int_repr.shape}, Int8 Weights:\n{weight_int_repr}\n")

    return weights_list

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for samples, labels in test_loader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

def load_ecg_data(data_dir):
    # Load and reshape data
    data_path = os.path.join(data_dir, 'data_1.npy')
    data = np.load(data_path)
    data = data.reshape(10000, 50, 1)

    # Load target data
    target_path = os.path.join(data_dir, 'data_3.npy')
    target = np.load(target_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=42
    )

    # Convert NumPy arrays to Tensors
    train_data_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_target_tensor = torch.tensor(y_train, dtype=torch.int64)
    test_data_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_target_tensor = torch.tensor(y_test, dtype=torch.int64)

    train_data_tensor = train_data_tensor.transpose(1, 2)
    test_data_tensor = test_data_tensor.transpose(1, 2)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_target_tensor)

    return train_dataset, test_dataset

def load_speech_command_datasets(data_dir, digit_labels):
    digit_files = {label: [] for label in digit_labels}
    for label in digit_labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.exists(label_dir):
            files = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir) if fname.endswith('.wav')]
            digit_files[label] = files
        else:
            print(f"Directory {label_dir} does not exist.")
    for label in digit_labels:
        print(f"Class '{label}' has {len(digit_files[label])} audio files.")
    n_mels_value = 32
    spectrograms = []
    labels = []
    for label_idx, label in enumerate(digit_labels):
        for file_path in digit_files[label]:
            y, sr = librosa.load(file_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels_value)
            S_dB = librosa.power_to_db(S, ref=np.max)
            if S_dB.shape[1] < 32:
                padding = 32 - S_dB.shape[1]
                S_dB = np.pad(S_dB, ((0, 0), (0, padding)), mode='constant')
            elif S_dB.shape[1] > 32:
                S_dB = S_dB[:, :32]
            S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
            spectrograms.append(S_dB)
            labels.append(label_idx)
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)
    train_spectrograms, test_spectrograms, train_labels, test_labels = train_test_split(
        spectrograms, labels, test_size=0.2, random_state=42, stratify=labels)

    class SpeechCommandDataset(Dataset):
        def __init__(self, spectrograms, labels):
            self.spectrograms = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.spectrograms[idx], self.labels[idx]

    train_dataset = SpeechCommandDataset(train_spectrograms, train_labels)
    test_dataset = SpeechCommandDataset(test_spectrograms, test_labels)

    return train_dataset, test_dataset

def load_dataset(folder_path):

    samples = []
    labels = []

    # 遍历指定文件夹中的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为csv文件
            if file.endswith('.csv'):
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 读取CSV文件中的数据
                data = pd.read_csv(file_path, header=None)
                # 将数据添加到样本列表
                samples.append(data.values)
                # 提取文件夹名称作为标签
                label = os.path.basename(root)
                labels.append(label)

    return samples, labels

class TimeSeriesDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples

        label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.labels = [label_mapping[label] for label in labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        x1 = torch.tensor(sample[0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        x2 = torch.tensor(sample[1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        x3 = torch.tensor(sample[2]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        x4 = torch.tensor(sample[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()

        return x1, x2, x3, x4, label