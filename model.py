import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 编码检测：解决中文文件读取问题
def detect_file_encoding(file_path):
    common_encodings = ['gbk', 'gb2312', 'utf-8', 'latin-1']
    for encoding in common_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.readline()
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("未检测到支持的编码格式，请检查文件完整性")

# 加载数据：目标列固定为“期末考试分数”
def load_and_preprocess_data():
    file_path = "student_data_adjusted_rounded.csv"
    encoding = detect_file_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    
    # 目标列：根据你的CSV实际列名设置为“期末考试分数”
    target = "期末考试分数"
    print(f"✅ 确认目标列：{target}")
    
    # 排除非特征列（学号、姓名），避免干扰模型
    exclude_cols = [target, "学号", "姓名"]
    X = df.drop([col for col in exclude_cols if col in df.columns], axis=1)
    y = df[target]  # 此时不会报KeyError，列名完全匹配
    
    # 编码分类特征（如性别、专业等）
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"🔧 分类特征：{categorical_cols}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"📊 数据加载完成：{X.shape[0]} 个样本，{X.shape[1]} 个特征")
    return X, y, df, target

# 训练并保存模型
def train_and_save_model():
    X, y, df, target = load_and_preprocess_data()
    
    # 划分训练集（80%）和测试集（20%）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 初始化随机森林模型（适配教育数据的参数）
    model = RandomForestRegressor(
        n_estimators=120,  # 树的数量
        max_depth=10,      # 最大深度，避免过拟合
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
    r2 = r2_score(y_test, y_pred)              # 决定系数（越接近1越好）
    print(f"\n📈 模型评估结果：")
    print(f"   - 平均绝对误差（MAE）：{mae:.2f} 分")
    print(f"   - 决定系数（R²）：{r2:.4f}")
    
    # 保存模型和关键配置（供Streamlit调用）
    joblib.dump(model, "student_score_model.pkl")  # 模型文件
    joblib.dump(X.columns.tolist(), "model_feature_names.pkl")  # 特征列表
    joblib.dump(df["专业"].unique().tolist(), "unique_majors.pkl")  # 专业列表
    joblib.dump(target, "target_column_name.pkl")  # 目标列名（避免后续硬编码）
    print(f"\n💾 模型文件已保存：")
    print(f"   - student_score_model.pkl（核心模型）")
    print(f"   - model_feature_names.pkl（特征列表）")
    print(f"   - unique_majors.pkl（专业列表）")

if __name__ == "__main__":
    train_and_save_model()
