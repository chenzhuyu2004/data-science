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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

# 设置全局字体为支持中文的字体
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (8, 6)  # 设置默认图大小

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

# 实现 DARTS 算法，加入早停机制和 TensorBoard 记录
class DARTS:
    def __init__(self, model, criterion, optimizer, scheduler, warmup_scheduler, device, history, accumulation_steps=1, patience=10, scheduler_type='default'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.device = device
        self.history = history
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.patience = patience
        self.scheduler_type = scheduler_type
        self.writer = SummaryWriter(log_dir=os.path.join(history.trial_dir, 'logs'))

    def search(self, train_loader, val_loader, epochs=100, warmup_epochs=10):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1} training', leave=False):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.squeeze())
                loss = loss / self.accumulation_steps
                loss.backward()

                self.current_step += 1
                if self.current_step % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.current_step = 0

                    if epoch < warmup_epochs:
                        self.warmup_scheduler.step()
                    else:
                        if self.scheduler_type == 'cyclic':
                            self.scheduler.step()

                train_loss += loss.item() * self.accumulation_steps

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

            current_lr = get_current_lr(self.optimizer)
            self.history.record(train_loss, val_loss, current_lr)
            logging.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            logging.info(f'Current LR: {current_lr}')

            # 记录到 TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)

            # 调度器的 step 调用
            if epoch >= warmup_epochs:
                if self.scheduler_type == 'reduce_on_plateau':
                    self.scheduler.step(val_loss)
                elif self.scheduler_type == 'cyclic':
                    pass  # 已在每个批次调用 step()
                else:
                    self.scheduler.step()

            # 早停机制
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.early_stop_counter = 0
                # 保存最佳模型
                os.makedirs(os.path.join(self.history.trial_dir, 'best_model'), exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.history.trial_dir, 'best_model', 'best_model.pth'))
                # 保存最佳超参数
                with open(os.path.join(self.history.trial_dir, 'best_params.json'), 'w') as f:
                    json.dump(self.history.params, f)
            else:
                self.history.early_stop_counter += 1
                if self.history.early_stop_counter >= self.patience:
                    logging.info(f'Early stopping after {epoch + 1} epochs.')
                    break

        self.writer.close()

# 定义 LossHistory 类
class LossHistory:
    def __init__(self, trial_dir, params):
        self.trial_dir = trial_dir
        os.makedirs(trial_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.lr_list = []
        self.params = params
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def record(self, train_loss, val_loss, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lr_list.append(lr)

    def save(self):
        np.savetxt(os.path.join(self.trial_dir, 'train_losses.csv'), self.train_losses, delimiter=",")
        np.savetxt(os.path.join(self.trial_dir, 'val_losses.csv'), self.val_losses, delimiter=",")
        np.savetxt(os.path.join(self.trial_dir, 'lr_list.csv'), self.lr_list, delimiter=",")

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
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop'])
    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 10)
    patience = trial.suggest_int('patience', 5, 15)
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
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        alpha = trial.suggest_float('alpha', 0.9, 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay)

    # 学习率调度器
    if trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'CyclicLR']) == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        scheduler_type = 'reduce_on_plateau'
    else:
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=500, mode='exp_range')
        scheduler_type = 'cyclic'

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs * len(train_loader))

    # 设置 LossHistory
    trial_dir = os.path.join(tuning_dir, f'trial_{trial.number}')
    history = LossHistory(trial_dir, hyperparams)
    history.best_val_loss = float('inf')
    history.early_stop_counter = 0

    # 设置 DARTS
    darts = DARTS(model, criterion, optimizer, scheduler, warmup_scheduler, device, history, accumulation_steps=accumulation_steps, patience=patience, scheduler_type=scheduler_type)

    # 执行 NAS 搜索
    darts.search(train_loader, val_loader, epochs=100, warmup_epochs=warmup_epochs)

    # 使用早停后的最佳验证损失作为目标值
    val_loss = history.best_val_loss

    # 保存最佳模型的超参数
    with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
        json.dump(hyperparams, f)

    history.save()

    return val_loss

# 定义获取当前学习率的函数
def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

# 定义绘制每个 trial 的损失曲线的函数
def plot_trials_loss_data(trial_dirs, project_dir, column):
    num_trials = len(trial_dirs)
    if num_trials == 0:
        logging.warning("没有可绘制的trial。")
        return

    cols = 5
    rows = (num_trials + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten() if rows > 1 else [axes]

    for i, trial_dir in enumerate(trial_dirs):
        train_losses_file = os.path.join(trial_dir, 'train_losses.csv')
        val_losses_file = os.path.join(trial_dir, 'val_losses.csv')
        params_file = os.path.join(trial_dir, 'best_params.json')

        if os.path.exists(train_losses_file) and os.path.exists(val_losses_file):
            train_losses = np.loadtxt(train_losses_file, delimiter=",")
            val_losses = np.loadtxt(val_losses_file, delimiter=",")

            axes[i].plot(train_losses, label='训练损失', color='blue', linestyle='-')
            axes[i].plot(val_losses, label='验证损失', color='orange', linestyle='--')

            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = json.load(f)
                key_params = {k: v for k, v in params.items() if k in ['cell_output_dims', 'learning_rate', 'activation']}
                title = f'Trial {i + 1}\n' + '\n'.join([f'{k}: {v}' for k, v in key_params.items()])
            else:
                title = f'Trial {i + 1}'

            axes[i].set_title(title, fontsize=8)
            axes[i].set_xlabel('Epochs', fontsize=10)
            axes[i].set_ylabel('Loss', fontsize=10)
            axes[i].legend(fontsize=8, framealpha=0.8)
            axes[i].grid(True)
        else:
            axes[i].set_title(f'Trial {i + 1}', fontsize=8)
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
    params_file = os.path.join(tuning_dir, f'trial_{best_trial.number}', 'best_params.json')

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

    # 复制最佳模型到指定目录
    destination_dir = os.path.join(BASE_DIR, 'Model_training_module', 'BayesianPytorchMLP')
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, f'best_model_{column}.pth')
    shutil.copy(best_model_path, destination_path)
    logging.info(f"Best model for {column} copied to {destination_path}")

    # 绘制最佳模型的损失曲线和学习率变化曲线
    best_trial_dir = os.path.join(tuning_dir, f'trial_{best_trial.number}')
    train_losses = np.loadtxt(os.path.join(best_trial_dir, 'train_losses.csv'), delimiter=",")
    val_losses = np.loadtxt(os.path.join(best_trial_dir, 'val_losses.csv'), delimiter=",")
    lr_list = np.loadtxt(os.path.join(best_trial_dir, 'lr_list.csv'), delimiter=",")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(train_losses, label='训练损失', color='blue', linestyle='-')
    ax1.plot(val_losses, label='验证损失', color='orange', linestyle='--')
    ax2.plot(lr_list, label='学习率', color='green', linestyle=':')

    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f'{column} - Best Model Loss and Learning Rate', fontsize=14)
    plt.savefig(os.path.join(main_training_dir, f'{column}_best_model_loss_lr.png'))
    plt.close()

    # 绘制残差分布图并添加数值标注
    residuals = y_val_column.cpu().numpy() - y_test_pred
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    median_res = np.median(residuals)

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title(f'{column} - 残差分布', fontsize=14)
    plt.xlabel('残差', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.text(0.05, 0.95, f'均值: {mean_res:.3f}', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.90, f'标准差: {std_res:.3f}', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.85, f'中位数: {median_res:.3f}', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(main_training_dir, f'{column}_residuals_distribution.png'))
    plt.close()

    # 绘制评估指标图
    plt.figure(figsize=(8, 6))
    error = np.abs(y_val_column.cpu().numpy() - y_test_pred)
    plt.scatter(y_val_column.cpu().numpy(), y_test_pred, edgecolors='k', s=error * 100, alpha=0.6)
    plt.plot([min(y_val_column.cpu().numpy()), max(y_val_column.cpu().numpy())],
             [min(y_val_column.cpu().numpy()), max(y_val_column.cpu().numpy())], 'k--', lw=2)
    plt.title(f'{column} - Best Model', fontsize=14)
    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)

    # 添加评估指标文本，使用背景框
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.90, f'MAE = {mae:.3f}', transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.85, f'MSE = {mse:.3f}', transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(main_training_dir, f'{column}_best_model_R2.png'))
    plt.close()

    # 绘制所有trial的损失曲线图
    trial_dirs = [os.path.join(tuning_dir, f'trial_{trial.number}') for trial in study.trials]
    plot_trials_loss_data(trial_dirs, main_training_dir, column)

# 绘制试验之间 R² 分布的柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(r2_scores.keys(), r2_scores.values(), color='lightblue', edgecolor='black')
plt.title('R² 分布', fontsize=14)
plt.xlabel('目标变量', fontsize=12)
plt.ylabel('R² 值', fontsize=12)
plt.xticks(rotation=45, fontsize=10)

# 添加数值标签，使用背景框
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(main_training_dir, 'r2_distribution.png'))
plt.close()