import os
import math
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
import logging

# 设置全局字体为支持中文的字体
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 配置 logging 模块
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 固定随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

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
    logging.error(f"数据读取失败: {e}")
    raise

# 预加载数据并创建 DataLoader
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# 定义 NASCell 和 NASNetwork
class NASCell(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(NASCell, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class NASNetwork(nn.Module):
    def __init__(self, input_size, cell_output_dims, activation):
        super(NASNetwork, self).__init__()
        self.cells = nn.ModuleList()
        current_features = input_size
        for out_dim in cell_output_dims:
            cell = NASCell(current_features, out_dim, activation)
            self.cells.append(cell)
            current_features = out_dim
        self.fc = nn.Linear(current_features, 1)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        x = self.fc(x)
        return x.squeeze()

# 实现 DARTS 算法，加入早停机制
class DARTS:
    def __init__(self, model, criterion, optimizer, scheduler, warmup_scheduler, device, history, accumulation_steps=1, patience=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.device = device
        self.history = history
        self.accumulation_steps = accumulation_steps
        self.current_step = 0  # 跟踪当前的累积步数
        self.patience = patience
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def search(self, train_loader, val_loader, epochs=50, warmup_epochs=10):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.squeeze())
                loss = loss / self.accumulation_steps  # 损失除以累积步数
                loss.backward()  # 累积梯度

                self.current_step += 1
                if self.current_step % self.accumulation_steps == 0:
                    self.optimizer.step()  # 更新模型参数
                    self.optimizer.zero_grad()  # 清空梯度
                    self.current_step = 0  # 重置累积步数

                    if epoch < warmup_epochs:
                        self.warmup_scheduler.step()
                    else:
                        self.scheduler.step()

                train_loss += loss.item() * self.accumulation_steps  # 累积损失

            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch.squeeze())
                    val_loss += loss.item()
                val_loss /= len(val_loader)

            self.history.record(train_loss, val_loss)
            logging.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # 早停机制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # 保存最佳模型
                os.makedirs(os.path.join(self.history.trial_dir, 'best_model'), exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.history.trial_dir, 'best_model', 'best_model.pth'))
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    logging.info(f'Early stopping after {epoch + 1} epochs.')
                    break

# 定义 LossHistory 类
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

# 提取超参数定义逻辑
def suggest_hyperparameters(trial):
    num_cells = trial.suggest_int('num_cells', 1, 5)
    cell_output_dims = []
    for i in range(num_cells):
        out_dim = trial.suggest_int(f'cell_{i}_out_dim', 16, 512, log=True)
        cell_output_dims.append(out_dim)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU'])
    accumulation_steps = trial.suggest_int('accumulation_steps', 1, 4)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 10)
    patience = trial.suggest_int('patience', 3, 10)
    return {
        'cell_output_dims': cell_output_dims,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'activation': activation,
        'accumulation_steps': accumulation_steps,
        'optimizer_name': optimizer_name,
        'warmup_epochs': warmup_epochs,
        'patience': patience
    }

# 定义 Optuna 目标函数，使用早停后的最佳验证损失
def objective(trial, train_loader, val_loader, device, tuning_dir, column):
    # 建议超参数
    hyperparams = suggest_hyperparameters(trial)
    cell_output_dims = hyperparams['cell_output_dims']
    learning_rate = hyperparams['learning_rate']
    weight_decay = hyperparams['weight_decay']
    activation_name = hyperparams['activation']
    accumulation_steps = hyperparams['accumulation_steps']
    optimizer_name = hyperparams['optimizer_name']
    warmup_epochs = hyperparams['warmup_epochs']
    patience = hyperparams['patience']

    # 定义激活函数
    if activation_name == 'ReLU':
        activation = nn.ReLU()
    elif activation_name == 'LeakyReLU':
        activation = nn.LeakyReLU()
    elif activation_name == 'ELU':
        activation = nn.ELU()

    # 构建 NASNetwork
    model = NASNetwork(input_size=X_train_tensor.shape[1],
                       cell_output_dims=cell_output_dims,
                       activation=activation).to(device)
    criterion = nn.MSELoss()

    # 定义优化器
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        alpha = trial.suggest_float('alpha', 0.9, 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs * len(train_loader))

    # 设置 LossHistory
    trial_dir = os.path.join(tuning_dir, f'trial_{trial.number}')
    history = LossHistory(trial_dir)

    # 设置 DARTS
    darts = DARTS(model, criterion, optimizer, scheduler, warmup_scheduler, device, history, accumulation_steps=accumulation_steps, patience=patience)

    # 执行 NAS 搜索
    darts.search(train_loader, val_loader, epochs=100, warmup_epochs=warmup_epochs)

    # 使用早停后的最佳验证损失作为目标值
    val_loss = darts.best_val_loss

    # 保存最佳模型的超参数
    with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
        json.dump(hyperparams, f)

    history.save()

    return val_loss

# 定义绘制每个 trial 的损失曲线的函数
def plot_trials_loss_data(trial_dirs, project_dir, column):
    num_trials = len(trial_dirs)
    if num_trials == 0:
        logging.warning("没有可绘制的trial。")
        return

    cols = 5
    rows = (num_trials + cols - 1) // cols
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
                key_params = {k: v for k, v in params.items() if k in ['cell_output_dims', 'learning_rate', 'activation']}
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

# 目标列列表
target_columns = y_train.columns.tolist()

# 存储所有试验的 R² 值
r2_scores = {}

# 对每个目标列进行训练和评估
for column in target_columns:
    logging.info(f"Training model for target variable: {column}")

    # 设置目录
    tuning_dir = os.path.join(main_training_dir, f'tuning_results_{column}')
    os.makedirs(tuning_dir, exist_ok=True)

    # 获取当前目标列
    y_train_column = y_train_tensor[:, target_columns.index(column)].to(device)
    y_val_column = y_val_tensor[:, target_columns.index(column)].to(device)

    # 修改训练和验证数据加载器以仅包含当前目标列
    train_dataset = TensorDataset(X_train_tensor, y_train_column)
    val_dataset = TensorDataset(X_val_tensor, y_val_column)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 设置 Optuna 研究，使用 TPESampler
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device, tuning_dir, column), n_trials=50)

    # 获取最佳试验并加载最佳模型
    best_trial = study.best_trial
    best_model_path = os.path.join(tuning_dir, f'trial_{best_trial.number}', 'best_model', 'best_model.pth')
    params_file = os.path.join(tuning_dir, f'trial_{best_trial.number}', 'params.json')

    with open(params_file, 'r') as f:
        params = json.load(f)

    cell_output_dims = params['cell_output_dims']
    best_activation = params['activation']

    if best_activation == 'ReLU':
        activation = nn.ReLU()
    elif best_activation == 'LeakyReLU':
        activation = nn.LeakyReLU()
    elif best_activation == 'ELU':
        activation = nn.ELU()

    best_model = NASNetwork(input_size=X_train_tensor.shape[1],
                            cell_output_dims=cell_output_dims,
                            activation=activation).to(device)
    best_model.load_state_dict(torch.load(best_model_path))

    # 评估最佳模型
    best_model.eval()
    with torch.no_grad():
        X_test_tensor = X_val_tensor.to(device)
        y_test_pred = best_model(X_test_tensor).detach().cpu().numpy()
    mae = mean_absolute_error(y_val_column.cpu().numpy(), y_test_pred)
    mse = mean_squared_error(y_val_column.cpu().numpy(), y_test_pred)
    r2 = r2_score(y_val_column.cpu().numpy(), y_test_pred)
    logging.info(f"{column} - MAE: {mae}, MSE: {mse}, R²: {r2}")
    r2_scores[column] = r2

    # 绘制评估指标图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_column.cpu().numpy(), y_test_pred, edgecolors=(0, 0, 0))
    plt.plot([min(y_val_column.cpu().numpy()), max(y_val_column.cpu().numpy())],
             [min(y_val_column.cpu().numpy()), max(y_val_column.cpu().numpy())], 'k--', lw=4)
    plt.title(f'{column} - Best Model')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f'MAE = {mae:.3f}', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.85, f'MSE = {mse:.3f}', transform=plt.gca().transAxes, fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(main_training_dir, f'{column}_best_model_R2.png'))
    plt.close()

    # 绘制试验损失数据图
    trial_dirs = [os.path.join(tuning_dir, f'trial_{trial.number}') for trial in study.trials]
    plot_trials_loss_data(trial_dirs, tuning_dir, column)

# 绘制试验之间 R² 分布的柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(r2_scores.keys(), r2_scores.values())
plt.title('R² 分布')
plt.xlabel('目标变量')
plt.ylabel('R² 值')
plt.xticks(rotation=45)

# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(main_training_dir, 'r2_distribution.png'))
plt.close()