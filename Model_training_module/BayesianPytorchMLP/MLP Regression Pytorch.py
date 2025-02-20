import os
import math
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置全局字体为支持中文的字体
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义数据目录
BASE_DIR = r"D:\pycharm\pythonProject"
DATA_DIR = os.path.join(BASE_DIR, 'Data_Source_Module')

# 读取数据并进行异常处理
try:
    X_train_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
    X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
except Exception as e:
    print(f"数据读取失败: {e}")
    raise

# 预加载数据并创建 DataLoader
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 定义MLP模型，支持不同的激活函数
class MLP(nn.Module):
    def __init__(self, input_size, layers, dropouts, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for i, (units, dropout) in enumerate(zip(layers, dropouts)):
            self.layers.append(nn.Linear(prev_size, units))
            activation_cls = getattr(nn, activation)
            self.layers.append(activation_cls())
            self.layers.append(nn.BatchNorm1d(units))
            self.layers.append(nn.Dropout(dropout))
            prev_size = units
        self.output = nn.Linear(prev_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x).squeeze()

# 定义损失历史记录类
class LossHistory:
    def __init__(self, trial_dir):
        self.trial_dir = trial_dir
        os.makedirs(trial_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []

    def record(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def save(self):
        np.savetxt(os.path.join(self.trial_dir, 'train_losses.csv'), self.train_losses, delimiter=",")
        np.savetxt(os.path.join(self.trial_dir, 'val_losses.csv'), self.val_losses, delimiter=",")

# 定义Optuna目标函数
def objective(trial, train_loader, y_train_column, val_loader, y_val_column):
    # 定义超参数
    num_layers = trial.suggest_int('num_layers', 1, 10)
    layers = []
    for i in range(num_layers):
        layers.append(trial.suggest_int(f'layer_{i}', 8, 1024, log=True))
    dropouts = []
    for i in range(num_layers):
        dropouts.append(trial.suggest_float(f'dropout_{i}', 0.0, 0.8))
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    patience = trial.suggest_int('patience', 10, 50)
    min_delta = trial.suggest_float('min_delta', 1e-5, 1e-1, log=True)
    activation = trial.suggest_categorical('activation', ['LeakyReLU', 'ReLU', 'ELU'])
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])

    # 构建模型
    model = MLP(input_size=X_train_tensor.shape[1], layers=layers, dropouts=dropouts, activation=activation).to(device)
    criterion = nn.MSELoss()
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    history = LossHistory(os.path.join(tuning_dir, f'trial_{trial.number}'))
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0

    for epoch in range(200):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch[:, target_columns.index(column)].to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        history.train_losses.append(epoch_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch[:, target_columns.index(column)].to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(val_loader)
        history.val_losses.append(epoch_val_loss)
        history.save()

        # 调整学习率
        scheduler.step()

        # 早停逻辑
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if trial.should_prune():
            raise optuna.TrialPruned()

    # 保存最佳模型
    if best_model_state is not None:
        os.makedirs(os.path.join(history.trial_dir, 'best_model'), exist_ok=True)
        torch.save(best_model_state, os.path.join(history.trial_dir, 'best_model', 'best_model.pth'))

    # 保存超参数
    with open(os.path.join(history.trial_dir, 'params.json'), 'w') as f:
        json.dump(trial.params, f)

    return best_val_loss

# 定义绘制每个trial的损失曲线的函数
def plot_trials_loss_data(trial_dirs, project_dir, column):
    num_trials = len(trial_dirs)
    if num_trials == 0:
        print("没有可绘制的trial。")
        return

    rows = math.ceil(num_trials / 5)
    cols = min(num_trials, 5)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes]

    for i, trial_dir in enumerate(trial_dirs):
        train_losses_file = os.path.join(trial_dir, 'train_losses.csv')
        val_losses_file = os.path.join(trial_dir, 'val_losses.csv')
        params_file = os.path.join(trial_dir, 'params.json')

        if os.path.exists(train_losses_file) and os.path.exists(val_losses_file):
            train_losses = np.loadtxt(train_losses_file, delimiter=",")
            val_losses = np.loadtxt(val_losses_file, delimiter=",")

            axes[i].plot(train_losses, label='训练损失', color='blue')
            axes[i].plot(val_losses, label='验证损失', color='orange')

            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = json.load(f)
                # 提取重要超参数
                key_params = {k: v for k, v in params.items() if
                              k in ['num_layers', 'learning_rate', 'weight_decay', 'patience', 'activation',
                                    'optimizer']}
                title = f'Trial {i + 1}\n' + '\n'.join([f'{k}: {v}' for k, v in key_params.items()])
            else:
                title = f'Trial {i + 1}'

            axes[i].set_title(title, fontsize=8)
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
            axes[i].grid(True)
        else:
            axes[i].set_title(f'Trial {i + 1}')
            axes[i].text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, f'{column}_trials_loss_plot.png'))
    plt.close()

# 主训练目录
main_training_dir = os.path.join(BASE_DIR, 'Model_training_module', 'BayesianPytorchMLP')
os.makedirs(main_training_dir, exist_ok=True)

# 针对每个目标变量进行训练
target_columns = y_train.columns.tolist()
for column in target_columns:
    print(f"Training model for target variable: {column}")

    y_train_column = y_train_tensor[:, target_columns.index(column)].to(device)
    y_val_column = y_val_tensor[:, target_columns.index(column)].to(device)

    tuning_dir = os.path.join(main_training_dir, f'tuning_results_{column}')
    os.makedirs(tuning_dir, exist_ok=True)

    # 设置Optuna研究
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, y_train_column, val_loader, y_val_column),
                   n_trials=30)

    # 获取最佳超参数
    best_params = study.best_params
    num_layers = best_params['num_layers']
    layers = [best_params[f'layer_{i}'] for i in range(num_layers)]
    dropouts = [best_params[f'dropout_{i}'] for i in range(num_layers)]
    activation = best_params['activation']
    optimizer_name = best_params['optimizer']

    best_model = MLP(input_size=X_train_tensor.shape[1],
                     layers=layers,
                     dropouts=dropouts,
                     activation=activation).to(device)

    # 加载最佳模型
    best_model_path = os.path.join(tuning_dir, f'trial_{study.best_trial.number}', 'best_model', 'best_model.pth')
    best_model.load_state_dict(torch.load(best_model_path))

    # 评估最佳模型
    best_model.eval()
    with torch.no_grad():
        X_test_tensor = X_val_tensor.to(device)
        y_test_pred = best_model(X_test_tensor).detach().cpu().numpy()
    mae = mean_absolute_error(y_val_column.cpu().numpy(), y_test_pred)
    mse = mean_squared_error(y_val_column.cpu().numpy(), y_test_pred)
    r2 = r2_score(y_val_column.cpu().numpy(), y_test_pred)
    print(f"{column} - MAE: {mae}, MSE: {mse}, R²: {r2}")

    # 绘制最佳模型预测图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_column.cpu().numpy(), y_test_pred, edgecolors=(0, 0, 0))
    plt.plot([min(y_val_column.cpu().numpy()), max(y_val_column.cpu().numpy())],
             [min(y_val_column.cpu().numpy()), max(y_val_column.cpu().numpy())], 'k--', lw=4)
    plt.title(f'{column} - Best Model')
    plt.xlabel('真实值')
    plt.ylabel('预测值')

    # 添加数值文本
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f'MAE = {mae:.3f}', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.85, f'MSE = {mse:.3f}', transform=plt.gca().transAxes, fontsize=10)

    plt.grid(True)
    plt.savefig(os.path.join(main_training_dir, f'{column}_best_model_R2.png'))
    plt.close()

    # 评估最佳模型在合并数据集上的表现
    combined_X = torch.cat((X_train_tensor, X_val_tensor), dim=0).to(device)
    combined_y = torch.cat((y_train_column, y_val_column), dim=0).to(device)
    combined_y_pred = best_model(combined_X).detach().cpu().numpy()
    combined_y_true = combined_y.cpu().numpy()
    r2 = r2_score(combined_y_true, combined_y_pred)
    mae = mean_absolute_error(combined_y_true, combined_y_pred)
    mse = mean_squared_error(combined_y_true, combined_y_pred)

    # 绘制合并数据集的R²图
    plt.figure(figsize=(6, 6))
    plt.scatter(combined_y_true, combined_y_pred, edgecolors=(0, 0, 0))
    plt.plot([min(combined_y_true), max(combined_y_true)],
             [min(combined_y_true), max(combined_y_true)], 'k--', lw=4)
    plt.title(f'{column} - Combined Data')
    plt.xlabel('真实值')
    plt.ylabel('预测值')

    # 添加数值文本
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f'MAE = {mae:.3f}', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.85, f'MSE = {mse:.3f}', transform=plt.gca().transAxes, fontsize=10)

    plt.grid(True)
    plt.savefig(os.path.join(main_training_dir, f'{column}_combined_R2.png'))
    plt.close()

    # 绘制每个trial的损失曲线
    trial_dirs = [os.path.join(tuning_dir, f'trial_{trial.number}') for trial in study.trials]
    plot_trials_loss_data(trial_dirs, tuning_dir, column)
