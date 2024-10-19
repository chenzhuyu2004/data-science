# 导入所需库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 定义文件路径
data_path = r'D:\pycharm\pythonProject\Data_Source_Module\train data.csv'
scaler_path = r'D:\pycharm\pythonProject\Data_Source_Module\scaler.pkl'
scaled_train_path = r'D:\pycharm\pythonProject\Data_Source_Module\X_train_scaled.csv'
scaled_test_path = r'D:\pycharm\pythonProject\Data_Source_Module\X_test_scaled.csv'
y_train_path = r'D:\pycharm\pythonProject\Data_Source_Module\y_train.csv'
y_test_path = r'D:\pycharm\pythonProject\Data_Source_Module\y_test.csv'

# 读取数据集
data = pd.read_csv(data_path)

# 显示数据前5行，以便检查是否正确导入
print("数据预览：")
print(data.head())

# 分割特征和目标变量
features = data.iloc[:, :8]  # 前8列作为特征
targets = data.iloc[:, 8:]    # 后3列作为目标变量

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# 初始化标准化器
scaler = StandardScaler()

# 对训练集进行标准化，并对测试集应用相同的转换
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存标准化器，以便在预测新数据时使用
joblib.dump(scaler, scaler_path)

print("数据集划分和标准化完成。标准化器已保存。")

# 将标准化后的数据转换为 DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features.columns, dtype='float64')
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features.columns, dtype='float64')

# 将目标变量转换为 DataFrame
y_train_df = pd.DataFrame(y_train.values, columns=targets.columns, dtype='float64')
y_test_df = pd.DataFrame(y_test.values, columns=targets.columns, dtype='float64')

# 导出标准化后的数据为 CSV 文件
X_train_scaled_df.to_csv(scaled_train_path, index=False)
X_test_scaled_df.to_csv(scaled_test_path, index=False)
y_train_df.to_csv(y_train_path, index=False)
y_test_df.to_csv(y_test_path, index=False)

print("标准化后的数据已导出至CSV文件。")
