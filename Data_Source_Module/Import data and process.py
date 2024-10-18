
# 导入所需库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 读取 CSV 文件
data_path = r'D:\pycharm\pythonProject\Data_Source_Module\train data.csv'
data = pd.read_csv(data_path)

# 显示数据前5行，以便检查是否正确导入
print(data.head())

# 分割特征和目标变量
features = data.iloc[:, :8]  # 前8列作为特征
targets = data.iloc[:, 8:]    # 后3列作为目标变量

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# 初始化标准化器并进行标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 保存标准化器
scaler_path = r'D:\pycharm\pythonProject\Data_Source_Module\scaler.pkl'
joblib.dump(scaler, scaler_path)

print("数据集划分和标准化完成。标准化器已保存。")

# 转换为 DataFrame 并导出标准化后的数据
def export_scaled_data(x_scaled, y, prefix):
    x_scaled_df = pd.DataFrame(x_scaled, columns=features.columns, dtype='float64')
    y_df = pd.DataFrame(y, dtype='float64')
    x_scaled_df.to_csv(f'D:\pycharm\pythonProject\Data_Source_Module\{prefix}_scaled.csv', index=False)
    y_df.to_csv(f'D:\pycharm\pythonProject\Data_Source_Module\{prefix}.csv', index=False)

export_scaled_data(x_train_scaled, y_train, 'x_train')
export_scaled_data(x_test_scaled, y_test, 'x_test')

print("标准化后的数据已导出至CSV文件。")

