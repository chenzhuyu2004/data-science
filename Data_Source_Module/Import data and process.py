import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    """
    主函数，用于读取数据、标准化特征和导出处理后的数据。
    """

    # 定义文件路径
    base_dir = os.getcwd()  # 获取当前工作目录
    data_path = os.path.join(base_dir,  'train data.csv')
    scaler_path = os.path.join(base_dir,  'scaler.pkl')
    scaled_train_path = os.path.join(base_dir,  'X_train_scaled.csv')
    scaled_test_path = os.path.join(base_dir,  'X_test_scaled.csv')
    y_train_path = os.path.join(base_dir,  'y_train.csv')
    y_test_path = os.path.join(base_dir,  'y_test.csv')

    # 读取数据集
    data = pd.read_csv(data_path)
    print("数据预览：")
    print(data.head())

    # 分割特征和目标变量
    features = data.iloc[:, :8]  # 前8列作为特征
    targets = data.iloc[:, 8:]  # 后3列作为目标变量

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
    export_scaled_data(X_train_scaled_df, X_test_scaled_df, y_train_df, y_test_df,
                       scaled_train_path, scaled_test_path, y_train_path, y_test_path)


def export_scaled_data(X_train_scaled_df, X_test_scaled_df, y_train_df, y_test_df,
                       scaled_train_path, scaled_test_path, y_train_path, y_test_path):
    """
    导出标准化后的数据为CSV文件。

    参数:
        X_train_scaled_df (DataFrame): 标准化后的训练特征
        X_test_scaled_df (DataFrame): 标准化后的测试特征
        y_train_df (DataFrame): 训练目标变量
        y_test_df (DataFrame): 测试目标变量
        scaled_train_path (str): 训练特征保存路径
        scaled_test_path (str): 测试特征保存路径
        y_train_path (str): 训练目标保存路径
        y_test_path (str): 测试目标保存路径
    """
    X_train_scaled_df.to_csv(scaled_train_path, index=False)
    X_test_scaled_df.to_csv(scaled_test_path, index=False)
    y_train_df.to_csv(y_train_path, index=False)
    y_test_df.to_csv(y_test_path, index=False)

    print("标准化后的数据已导出至CSV文件。")


if __name__ == "__main__":
    main()
