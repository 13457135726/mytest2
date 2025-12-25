import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os

# -------------------------- 基础配置 --------------------------
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="📊",
    layout="wide"
)

# 强制Matplotlib使用支持中文的字体（解决中文显示问题）
# 优先使用系统中常见的UTF-8编码中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = 'white'

# -------------------------- 工具函数（强制UTF-8） --------------------------
def safe_load_csv_utf8(file_path):
    """强制以UTF-8编码加载CSV文件（含BOM处理）"""
    if not os.path.exists(file_path):
        st.error(f"数据文件不存在：{os.path.abspath(file_path)}")
        return None
    
    try:
        # 优先尝试UTF-8（含BOM），强制指定编码
        df = pd.read_csv(
            file_path,
            encoding='utf-8-sig',  # utf-8-sig自动处理BOM
            on_bad_lines='skip'    # 跳过格式错误行
        )
        return df
    except UnicodeDecodeError:
        st.error(f"文件 {file_path} 不是UTF-8编码！请将文件转换为UTF-8编码后重试。")
        return None
    except Exception as e:
        st.error(f"加载CSV失败：{str(e)}")
        return None

def safe_load_model_utf8(model_path):
    """安全加载模型文件（强制UTF-8路径处理）"""
    # 确保路径为UTF-8编码字符串
    model_path = os.fsencode(model_path).decode('utf-8')
    
    if not os.path.exists(model_path):
        st.warning(f"模型文件不存在：{os.path.abspath(model_path)}")
        return None
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"加载模型 {os.path.basename(model_path)} 失败：{str(e)}")
        return None

# -------------------------- 数据与模型加载（全程UTF-8） --------------------------
@st.cache_data(ttl=3600)
def load_data_and_models():
    """加载数据和模型（强制UTF-8编码）"""
    # 1. 加载数据（强制UTF-8）
    data_file = "student_data_adjusted_rounded.csv"
    df = safe_load_csv_utf8(data_file)
    if df is None:
        return None, None, None, None, None

    # 2. 列名标准化（UTF-8编码下的清理）
    df.columns = df.columns.str.strip()  # 仅去除首尾空格，不做编码转换
    # 统一列名格式（仅字符替换，不涉及编码）
    df.columns = df.columns.str.replace('（小时）', '(小时)', regex=False)
    df.columns = df.columns.str.replace('（', '(', regex=False)
    df.columns = df.columns.str.replace('）', ')', regex=False)

    # 3. 验证必要列
    required_cols = ['学号', '性别', '专业', '每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"数据文件缺少必要列：{missing_cols}")
        return None, None, None, None, None

    # 4. 数据清洗（仅数值转换，无编码操作）
    df = df[required_cols].dropna()
    numeric_cols = ['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    # 5. 加载模型（强制UTF-8路径）
    reg_model = safe_load_model_utf8("linear_regression_model.pkl")
    clf_model = safe_load_model_utf8("random_forest_clf.pkl")
    clf_feature_cols = safe_load_model_utf8("clf_feature_cols.pkl")
    encoder = safe_load_model_utf8("onehot_encoder.pkl")

    return df, reg_model, clf_model, clf_feature_cols, encoder

# -------------------------- 测试加载（可选） --------------------------
if __name__ == "__main__":
    df, reg_model, clf_model, clf_feature_cols, encoder = load_data_and_models()
    if df is not None:
        st.success("数据加载成功！")
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.error("数据加载失败，请检查文件编码和路径！")
