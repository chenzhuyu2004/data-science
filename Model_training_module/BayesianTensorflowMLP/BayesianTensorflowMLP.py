import math
import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l1_l2

# 设置全局字体为支持中文的字体
mpl.rcParams['font.family'] = 'Microsoft YaHei'  # 或者使用其他支持中文的字体
mpl.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题

# 定义数据目录
BASE_DIR = r"D:\pycharm\pythonProject"
DATA_DIR = os.path.join(BASE_DIR, 'Data_Source_Module')

# 读取数据
try:
    X_train_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
    X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
except Exception as e:
    print(f"数据读取失败: {e}")

# 启用全局混合精度策略
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


def create_dataset(X_data: np.ndarray, y_data: np.ndarray, batch_size: int = 64) -> tf.data.Dataset:
    """
    创建TensorFlow数据集。

    :param X_data: 输入特征数据
    :param y_data: 标签数据
    :param batch_size: 批量大小
    :return: TensorFlow数据集
    """
    return tf.data.Dataset.from_tensor_slices((X_data, y_data)).batch(batch_size).prefetch(AUTOTUNE)


def build_model(hp) -> keras.Sequential:
    """
    构建神经网络模型。

    :param hp: 超参数调优对象
    :return: 构建的模型
    """
    model = keras.Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))

    for i in range(hp.Int('num_layers', 1, 15)):
        l1_param = hp.Float(f'l1_{i}', min_value=0.0, max_value=1e-5, step=1e-6)
        l2_param = hp.Float(f'l2_{i}', min_value=0.0, max_value=1e-5, step=1e-6)

        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=1024, step=32),
                        activation='relu',
                        kernel_regularizer=l1_l2(l1=l1_param, l2=l2_param)))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.06, max_value=0.5, step=0.02)))

    model.add(Dense(1, dtype='float32'))
    optimizer = AdamW(learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, sampling='log'),
                      clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


class LossHistory(tf.keras.callbacks.Callback):
    """记录训练过程中的损失和验证损失。"""

    def __init__(self, trial_dir: str):
        super().__init__()
        self.trial_dir = trial_dir
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch: int, logs=None):
        """在每个epoch结束时保存损失数据。"""
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.save_losses()

    def save_losses(self):
        """保存损失数据到CSV文件。"""
        np.savetxt(os.path.join(self.trial_dir, 'losses.csv'), self.losses, delimiter=",")
        np.savetxt(os.path.join(self.trial_dir, 'val_losses.csv'), self.val_losses, delimiter=",")


class AdaptiveTuner(BayesianOptimization):
    """自适应超参数调优器。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_results = []

    def run_trial(self, trial, *args, **kwargs):
        """运行一个超参数调优试验。"""
        trial_id = trial.trial_id
        trial_dir = os.path.join(tuning_dir, f'trial_{trial_id}')
        os.makedirs(trial_dir, exist_ok=True)

        history = LossHistory(trial_dir)
        callbacks = [
            history,
            ModelCheckpoint(filepath=os.path.join(trial_dir, f'trial_{trial_id}_MLP_best_model.keras'),
                            monitor='val_loss', save_best_only=True)
        ]

        kwargs['callbacks'] = callbacks
        result = super().run_trial(trial, *args, **kwargs)
        self.trial_results.append((trial, history.val_losses[-1] if history.val_losses else None))
        return result


# 定义主目录和子目录
main_training_dir = os.path.join(BASE_DIR, 'Model_training_module', 'BayesianTensorflowMLP')
os.makedirs(main_training_dir, exist_ok=True)

# 目标变量列名
target_columns = y_train.columns.tolist()

# 回调函数列表
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-6)
]

# 针对每个目标变量进行训练
for column in target_columns:
    print(f"Training model for target variable: {column}")

    y_train_column = y_train[column].values
    y_test_column = y_test[column].values

    dataset_train_column = create_dataset(X_train_scaled, y_train_column)
    dataset_val_column = create_dataset(X_test_scaled, y_test_column)

    tuning_dir = os.path.join(main_training_dir, f'tuning_results_{column}')
    os.makedirs(tuning_dir, exist_ok=True)

    # 超参数搜索
    tuner = AdaptiveTuner(
        build_model,
        objective='val_loss',
        max_trials=30,
        executions_per_trial=1,
        directory=tuning_dir,
        project_name=f'optimized_{column}',
        overwrite=True
    )

    # 搜索超参数
    try:
        tuner.search(dataset_train_column, epochs=100, validation_data=dataset_val_column, callbacks=callbacks)
    except Exception as e:
        print(f"{column} - 超参数搜索失败: {e}")

    # 找到最佳 trial
    trial_dirs = [d for d in os.listdir(tuning_dir) if d.startswith('trial_')]
    trial_dirs.sort(key=lambda x: int(x.split('_')[1]))

    best_val_loss = float('inf')
    best_trial_id = None

    for trial_dir in trial_dirs:
        val_losses_file = os.path.join(tuning_dir, trial_dir, 'val_losses.csv')
        if os.path.exists(val_losses_file):
            val_losses = np.loadtxt(val_losses_file, delimiter=",")
            if len(val_losses) > 0:
                min_val_loss = np.min(val_losses)
                if min_val_loss < best_val_loss:
                    best_val_loss = min_val_loss
                    best_trial_id = trial_dir.split('_')[1]

    # 如果找到最佳模型，保存最终模型路径
    if best_trial_id is not None:
        best_model_path = os.path.join(tuning_dir, f'trial_{best_trial_id}',
                                       f'trial_{best_trial_id}_MLP_best_model.keras')

        if os.path.exists(best_model_path):
            final_model_path = os.path.join(main_training_dir, f'final_best_MLP_model_{column}.keras')
            shutil.copy(best_model_path, final_model_path)

            best_model = keras.models.load_model(final_model_path)

            # 使用最佳模型进行预测并绘制结果
            y_pred_column = best_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test_column, y_pred_column)
            mse = mean_squared_error(y_test_column, y_pred_column)
            r2 = r2_score(y_test_column, y_pred_column)

            print(f"{column} - MAE: {mae}, MSE: {mse}, R²: {r2}")

            # 绘制最佳模型的 R² 图
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test_column, y_pred_column, edgecolors=(0, 0, 0))
            plt.plot([min(y_test_column), max(y_test_column)], [min(y_test_column), max(y_test_column)], 'k--', lw=4)
            plt.title(f'{column} - Best Model R²: {r2:.3f}')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.grid(True)
            plt.savefig(os.path.join(main_training_dir, f'{column}_best_model_R2.png'))
            plt.close()

            # 绘制合并训练集和测试集后的 R² 图
            combined_X = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
            combined_y = np.concatenate([y_train_column, y_test_column], axis=0)
            combined_y_pred = np.concatenate([best_model.predict(X_train_scaled), y_pred_column])

            plt.figure(figsize=(6, 6))
            plt.scatter(combined_y, combined_y_pred, edgecolors=(0, 0, 0))
            plt.plot([min(combined_y), max(combined_y)], [min(combined_y), max(combined_y)], 'k--', lw=4)
            plt.title(f'Combined Data R²: {r2_score(combined_y, combined_y_pred):.3f}')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.grid(True)
            plt.savefig(os.path.join(main_training_dir, f'combined_{column}_R2.png'))
            plt.close()

        else:
            print(f"{column} - 找不到最佳模型文件.")
    else:
        print(f"{column} - 未找到最佳试验 ID.")

    # 清理模型信息以准备下一个目标变量
    tf.keras.backend.clear_session()
    keras.backend.clear_session()
    del tuner  # 删除调优器对象


def plot_trials_loss_data(trial_dirs: list, project_dir: str, column: str):
    """
    绘制每个 trial 的训练损失和验证损失曲线。

    :param trial_dirs: 试验目录列表
    :param project_dir: 项目目录
    :param column: 目标变量名称
    """
    num_trials = len(trial_dirs)
    if num_trials == 0:
        print("没有可绘制的 trial。")
        return

    rows = math.ceil(num_trials / 5)  # 自动调整行数
    cols = min(num_trials, 5)  # 列数不超过 5

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes]

    for trial_idx in range(num_trials):
        trial_dir = trial_dirs[trial_idx]
        losses_file = os.path.join(project_dir, trial_dir, 'losses.csv')
        val_losses_file = os.path.join(project_dir, trial_dir, 'val_losses.csv')

        if os.path.exists(losses_file) and os.path.exists(val_losses_file):
            losses = np.loadtxt(losses_file, delimiter=",")
            val_losses = np.loadtxt(val_losses_file, delimiter=",")
            axes[trial_idx].plot(losses, label='训练损失', color='blue')
            axes[trial_idx].plot(val_losses, label='验证损失', color='orange')
            axes[trial_idx].set_title(trial_dir)
            axes[trial_idx].set_xlabel('Epochs')
            axes[trial_idx].set_ylabel('Loss')
            axes[trial_idx].legend()
            axes[trial_idx].grid(True)
        else:
            axes[trial_idx].set_title(trial_dir)
            axes[trial_idx].text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center')
            axes[trial_idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, f'{column}_trials_loss_plot.png'))
    plt.close()


# 绘制每个目标变量的损失数据
for column in target_columns:
    tuning_dir = os.path.join(main_training_dir, f'tuning_results_{column}')
    trial_dirs = [d for d in os.listdir(tuning_dir) if d.startswith('trial_')]
    plot_trials_loss_data(trial_dirs, tuning_dir, column)

