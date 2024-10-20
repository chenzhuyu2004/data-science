import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import optuna
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# 设置基础目录
BASE_DIR = r"D:\pycharm\pythonProject"
DATA_DIR = os.path.join(BASE_DIR, 'Data_Source_Module')

# 配置字典
config = {
    'training': {
        'early_stopping_threshold': 1e-4,
        'initial_patience': 10,
        'max_epochs': 100,
        'max_epochs_increment': 10,
        'min_batch_size': 16,
        'initial_batch_size': 64,
    },
    'hyperparameters': {
        'model': {
            'num_layers': (1, 16),
            'units': (32, 1024),
            'l1': (0.0, 1e-5),
            'l2': (0.0, 1e-5),
            'dropout_rate': (0.06, 0.5),
            'learning_rate': (1e-6, 1e-2),
        },
        'optimizer': {
            'type': ['AdamW', 'SGD'],
            'momentum': (0.0, 0.9),  # 仅适用于SGD
        },
        'scheduler': {
            'mode': 'min',
            'factor': 0.1,
            'patience': 5,
            'threshold': 1e-4,
        }
    }
}


# 数据读取
try:
    X_train_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
    X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
except Exception as e:
    print(f"数据读取失败: {e}")


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.labels[idx]


# 定义动态的多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self, input_size: int, num_layers: int, units: int, l1: float, l2: float, dropout_rate: float):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, units), nn.ReLU(), nn.Dropout(dropout_rate)]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(units, 1))  # 输出层
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# 批量大小管理类
class BatchSizeManager:
    def __init__(self, initial_batch_size: int, min_batch_size: int):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.last_best_val_loss = float('inf')

    def adjust(self, val_loss: float) -> int:
        if val_loss > self.last_best_val_loss:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        self.last_best_val_loss = val_loss
        return self.current_batch_size


# Optuna优化目标函数
def objective(trial: optuna.Trial, target_variable: str) -> float:
    batch_manager = BatchSizeManager(config['training']['initial_batch_size'], config['training']['min_batch_size'])

    model_params = {
        'num_layers': trial.suggest_int('num_layers', *config['hyperparameters']['model']['num_layers']),
        'units': trial.suggest_int('units', *config['hyperparameters']['model']['units']),
        'l1': trial.suggest_float('l1', *config['hyperparameters']['model']['l1']),
        'l2': trial.suggest_float('l2', *config['hyperparameters']['model']['l2']),
        'dropout_rate': trial.suggest_float('dropout_rate', *config['hyperparameters']['model']['dropout_rate']),
    }
    learning_rate = trial.suggest_float('learning_rate', *config['hyperparameters']['model']['learning_rate'], log=True)

    optimizer_type = trial.suggest_categorical('optimizer_type', config['hyperparameters']['optimizer']['type'])
    momentum = trial.suggest_float('momentum',
                                   *config['hyperparameters']['optimizer']['momentum']) if optimizer_type == 'SGD' else 0

    model = MLP(X_train_scaled.shape[1], **model_params)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) if optimizer_type == 'AdamW' else optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum)
    scaler = torch.amp.GradScaler('cuda')

    scheduler = ReduceLROnPlateau(optimizer, mode=config['hyperparameters']['scheduler']['mode'],
                                  factor=config['hyperparameters']['scheduler']['factor'],
                                  patience=config['hyperparameters']['scheduler']['patience'],
                                  threshold=config['hyperparameters']['scheduler']['threshold'])

    log_dir = os.path.join(main_training_dir, f'tuning_results_{target_variable}_trial_{trial.number}')
    writer = SummaryWriter(log_dir)

    y_train_column = y_train[target_variable].values

    # 交叉验证
    kf = KFold(n_splits=5)
    val_losses = []

    for train_index, val_index in kf.split(X_train_scaled):
        train_features, val_features = X_train_scaled.iloc[train_index], X_train_scaled.iloc[val_index]
        train_labels, val_labels = y_train_column[train_index], y_train_column[val_index]

        train_dataset = CustomDataset(train_features, pd.DataFrame(train_labels))
        val_dataset = CustomDataset(val_features, pd.DataFrame(val_labels))

        train_loader = DataLoader(train_dataset, batch_size=batch_manager.current_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_manager.current_batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        dynamic_epochs = config['training']['max_epochs']

        # 训练模型
        for epoch in range(dynamic_epochs):
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                optimizer.zero_grad()

                # 启用混合精度
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                # 使用 GradScaler 来缩放损失
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            val_predictions = []
            val_labels_list = []
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                    # 使用混合精度
                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_features)

                    outputs = outputs.float()  # 确保输出为原始精度
                    val_loss += criterion(outputs, batch_labels).item()
                    val_predictions.append(outputs.cpu().numpy())
                    val_labels_list.append(batch_labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_predictions = np.concatenate(val_predictions)
            val_labels_list = np.concatenate(val_labels_list)

            # 计算额外的评估指标
            mae = mean_absolute_error(val_labels_list, val_predictions)
            r2 = r2_score(val_labels_list, val_predictions)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MAE/val', mae, epoch)
            writer.add_scalar('R2/val', r2, epoch)

            # 更新学习率调度器
            scheduler.step(val_loss)

            # 动态调整批量大小
            batch_manager.adjust(val_loss)
            train_loader = DataLoader(train_dataset, batch_size=batch_manager.current_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_manager.current_batch_size, shuffle=False)

            # 动态调整训练轮数
            if val_loss < best_val_loss - config['training']['early_stopping_threshold']:
                best_val_loss = val_loss
                patience_counter = 0
                dynamic_epochs += config['training']['max_epochs_increment']
            else:
                patience_counter += 1
                if patience_counter >= config['training']['initial_patience']:
                    break  # 只在这里停止当前折的训练

        val_losses.append(best_val_loss)

    # 在所有折结束后输出一次最佳验证损失
    overall_best_val_loss = np.min(val_losses)
    print(f"最佳验证损失为: {overall_best_val_loss}")

    writer.close()
    trial.set_user_attr('model_state_dict', model.state_dict())
    return overall_best_val_loss


# 创建主训练目录
main_training_dir = os.path.join(BASE_DIR, 'Model_training_module', 'Model_1', 'train_dir')
os.makedirs(main_training_dir, exist_ok=True)

target_columns = y_train.columns.tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for column in target_columns:
    print(f"Training model for target variable: {column}")

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, column), n_trials=100)

    print(f"最佳超参数 for {column}:", study.best_params)

    best_model_state_dict = study.best_trial.user_attrs['model_state_dict']
    best_model = MLP(X_train_scaled.shape[1], study.best_params['num_layers'],
                     study.best_params['units'], study.best_params['l1'],
                     study.best_params['l2'], study.best_params['dropout_rate'])
    best_model.load_state_dict(best_model_state_dict)
    torch.save(best_model.state_dict(), os.path.join(main_training_dir, f'best_model_{column}.pt'))


