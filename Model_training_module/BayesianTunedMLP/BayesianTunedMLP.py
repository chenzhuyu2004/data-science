#  导入所需库
import math
import os
import shutil
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

# 读取数据
data_dir = 'D:\\pycharm\\pythonProject\\Data_Source_Module'
try:
    X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values[:, 0].squeeze()
    X_test_scaled = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values[:, 0].squeeze()
except Exception as e:
    print(f"数据读取失败: {e}")

# 启用全局混合精度策略
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 准备数据集的函数
def create_dataset(X_data, y_data, batch_size=64):
    return tf.data.Dataset.from_tensor_slices((X_data, y_data)).batch(batch_size).prefetch(AUTOTUNE)

# 创建训练集和验证集
dataset_train = create_dataset(X_train_scaled, y_train)
dataset_val = create_dataset(X_test_scaled, y_test)

# 定义模型构建函数
def build_model(hp):
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

# 定义回调函数用于记录和保存损失
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, trial_dir):
        super().__init__()
        self.trial_dir = trial_dir
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.save_losses()

    def save_losses(self):
        np.savetxt(os.path.join(self.trial_dir, 'losses.csv'), self.losses, delimiter=",")
        np.savetxt(os.path.join(self.trial_dir, 'val_losses.csv'), self.val_losses, delimiter=",")

# 自定义 AdaptiveTuner 类
class AdaptiveTuner(BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_results = []

    def run_trial(self, trial, *args, **kwargs):
        trial_id = trial.trial_id
        trial_dir = os.path.join(tuning_dir, f'trial_{trial_id}')
        os.makedirs(trial_dir, exist_ok=True)

        history = LossHistory(trial_dir)
        callbacks = [history, ModelCheckpoint(filepath=os.path.join(trial_dir, f'trial_{trial_id}_MLP_best_model.keras'),
                                               monitor='val_loss', save_best_only=True)]

        kwargs['callbacks'] = callbacks
        result = super().run_trial(trial, *args, **kwargs)

        self.trial_results.append((trial, history.val_losses[-1] if history.val_losses else None))

        return result

# 定义主目录和子目录
main_training_dir = 'D:\\pycharm\\pythonProject\\Model_training_module\\BayesianTunedMLP\\new_training_directory'
if not os.path.exists(main_training_dir):
    os.makedirs(main_training_dir)

# 定义项目的子目录，用来保存每次的训练结果和优化信息
tuning_dir = os.path.join(main_training_dir, 'tuning_results')

# 回调函数列表
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-6)
]

# 超参数搜索
tuner = AdaptiveTuner(
    build_model,
    objective='val_loss',
    max_trials=30,
    executions_per_trial=1,
    directory=tuning_dir,
    project_name='optimized_single_output_regression',
    overwrite=True
)

# 搜索超参数
tuner.search(dataset_train, epochs=100, validation_data=dataset_val, callbacks=callbacks)

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
    best_model_path = os.path.join(tuning_dir, f'trial_{best_trial_id}', f'trial_{best_trial_id}_MLP_best_model.keras')

    if os.path.exists(best_model_path):
        final_model_path = os.path.join(main_training_dir, 'final_best_MLP_model.keras')
        shutil.copy(best_model_path, final_model_path)

        best_model = keras.models.load_model(final_model_path)

        # 使用最佳模型进行预测并绘制结果
        y_pred = best_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f'MAE: {mae}, MSE: {mse}')

        r2 = r2_score(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
        plt.title(f'R²: {r2:.3f}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True)
        plt.show()
    else:
        print(f"模型文件不存在: {best_model_path}")
else:
    print("未找到最佳 trial 的模型文件。")

# 绘制每个 trial 的训练损失和验证损失曲线
def plot_trials_loss_data(trial_dirs, project_dir):
    num_trials = len(trial_dirs)
    if num_trials == 0:
        print("No trials to plot.")
        return

    cols = 5
    rows = math.ceil(num_trials / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes]

    for trial_idx in range(num_trials):
        trial_dir = trial_dirs[trial_idx]
        losses_file = os.path.join(project_dir, trial_dir, 'losses.csv')
        val_losses_file = os.path.join(project_dir, trial_dir, 'val_losses.csv')

        if os.path.exists(losses_file) and os.path.exists(val_losses_file):
            losses = np.loadtxt(losses_file, delimiter=",")
            val_losses = np.loadtxt(val_losses_file, delimiter=",")

            axes[trial_idx].plot(losses, label='Training Loss', color='blue')
            axes[trial_idx].plot(val_losses, label='Validation Loss', color='orange')
            axes[trial_idx].set_title(trial_dir)
            axes[trial_idx].set_xlabel('Epochs')
            axes[trial_idx].set_ylabel('Loss')
            axes[trial_idx].legend()
            axes[trial_idx].grid(True)
        else:
            axes[trial_idx].set_title(trial_dir)
            axes[trial_idx].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            axes[trial_idx].axis('off')

    plt.tight_layout()
    plt.show()

# 绘制所有 trial 的损失数据
plot_trials_loss_data(trial_dirs, tuning_dir)

X_combined = np.vstack((X_train_scaled, X_test_scaled))
y_combined = np.hstack((y_train, y_test))

# 使用最佳模型进行预测
y_pred_combined = best_model.predict(X_combined)

# 计算合并数据集上的 R² 值
r2_combined = r2_score(y_combined, y_pred_combined)

# 绘制 R² 图
plt.figure(figsize=(6, 6))
plt.scatter(y_combined, y_pred_combined, edgecolors=(0, 0, 0))
plt.plot([min(y_combined), max(y_combined)], [min(y_combined), max(y_combined)], 'k--', lw=4)
plt.title(f'R² (combined): {r2_combined:.3f}')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.grid(True)
plt.show()

print(f'Combined R²: {r2_combined}')
