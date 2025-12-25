import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import chardet
import re

# -------------------------- 1. 基础配置 --------------------------
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="📊",
    layout="wide"
)

# 中文显示修复
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.markersize'] = 6

# -------------------------- 核心修改1：等级图片仅用纯文件名（无路径） --------------------------
GRADE_IMAGE_MAP = {
    "优秀": ["优秀.PNG", "优秀.png"],  # 兼容大小写
    "良好": ["良好.PNG", "良好.png"],
    "及格": ["及格.PNG", "及格.png"],
    "不及格": ["未及格.PNG", "不及格.PNG", "未及格.png", "不及格.png"]  # 包含"未及格"文件名
}

# -------------------------- 2. 核心工具函数 --------------------------
def detect_file_encoding(file_path):
    if not os.path.exists(file_path):
        st.warning(f"文件不存在：{file_path}")
        return 'utf-8-sig'
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10240)
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8-sig'
        return 'gbk' if encoding.lower() in ['gb2312', 'gbk'] else encoding
    except Exception as e:
        st.error(f"编码检测失败：{str(e)}")
        return 'utf-8-sig'

def safe_load_model(model_path):
    if not os.path.exists(model_path):
        st.warning(f"模型缺失：{os.path.basename(model_path)}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"加载{os.path.basename(model_path)}失败：{str(e)}")
        return None

# -------------------------- 核心修改2：获取等级图片仅用纯文件名判断 --------------------------
def get_grade_image(grade):
    if grade not in GRADE_IMAGE_MAP:
        return None
    # 仅遍历纯文件名，判断当前目录是否存在
    for img_filename in GRADE_IMAGE_MAP[grade]:
        if os.path.exists(img_filename):  # 直接用纯文件名判断，无路径
            return img_filename
    # 提示时仅显示纯文件名
    st.warning(f"未找到{grade}等级图片，可放置以下任一纯文件名到当前目录：{GRADE_IMAGE_MAP[grade]}")
    return None

def find_matching_column(df_columns, target_keywords):
    target_keywords = [kw.lower() for kw in target_keywords]
    for col in df_columns:
        col_lower = col.lower().strip()
        if any(kw in col_lower for kw in target_keywords):
            return col
    return None

# -------------------------- 3. 数据与模型加载 --------------------------
@st.cache_data(ttl=3600, show_spinner="加载数据与模型中...")
def load_data_and_models():
    csv_path = "student_data_adjusted_rounded.csv"  # 纯文件名
    df = None
    if os.path.exists(csv_path):
        try:
            encoding = detect_file_encoding(csv_path)
            df = pd.read_csv(csv_path, encoding=encoding)
            df.columns = [col.strip() for col in df.columns]
            original_columns = df.columns.tolist()

            column_mapping = {
                '学号': ['学号', '学生编号', 'id'],
                '性别': ['性别', '男/女'],
                '专业': ['专业', '学科', '专业名称'],
                '每周学习时长': ['每周学习时长', '每周学习时间', '学习时长/周', '周学习时长'],
                '上课出勤率': ['上课出勤率', '出勤率', '上课出勤比例'],
                '期中考试分数': ['期中考试分数', '期中分数', '期中成绩'],
                '作业完成率': ['作业完成率', '作业完成比例', '作业完成度'],
                '期末考试分数': ['期末考试分数', '期末分数', '期末成绩']
            }

            matched_columns = {}
            missing_standard_cols = []
            for standard_col, keywords in column_mapping.items():
                matched_col = find_matching_column(df.columns, keywords)
                if matched_col:
                    matched_columns[standard_col] = matched_col
                    df.rename(columns={matched_col: standard_col}, inplace=True)
                else:
                    missing_standard_cols.append(standard_col)

            for missing_col in missing_standard_cols:
                if missing_col == '每周学习时长':
                    df[missing_col] = np.random.uniform(10, 25, len(df))
                    st.warning(f"CSV缺少'{missing_col}'列，已用10-25小时随机值填充")
                elif '分数' in missing_col:
                    df[missing_col] = np.random.uniform(60, 85, len(df))
                    st.warning(f"CSV缺少'{missing_col}'列，已用60-85分随机值填充")
                elif '率' in missing_col:
                    df[missing_col] = np.random.uniform(0.7, 0.95, len(df))
                    st.warning(f"CSV缺少'{missing_col}'列，已用0.7-0.95随机值填充")
                else:
                    if missing_col == '性别':
                        df[missing_col] = np.random.choice(['男', '女'], len(df))
                    elif missing_col == '专业':
                        df[missing_col] = '大数据管理'
                    st.warning(f"CSV缺少'{missing_col}'列，已用默认值填充")

            numeric_cols = ['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

            st.success(f"✅ 数据加载完成：{len(df)}条记录")
        except Exception as e:
            st.error(f"CSV加载失败：{str(e)}")
            return None, None, None, None, None
    else:
        st.error(f"未找到CSV文件：{csv_path}（请将文件放在代码同一目录）")
        return None, None, None, None, None

    # 模型文件均用纯文件名加载
    reg_model = safe_load_model("student_score_model.pkl")
    model_features = safe_load_model("model_feature_names.pkl")
    unique_majors = safe_load_model("unique_majors.pkl") or (df['专业'].unique() if df is not None else ["大数据管理"])
    target_col = safe_load_model("target_column_name.pkl") or "期末考试分数"

    return df, reg_model, model_features, unique_majors, target_col

# 初始化变量
student_df, reg_model, model_features, unique_majors, target_col = load_data_and_models()

# -------------------------- 4. 模型输入构建函数 --------------------------
def build_reg_input(input_data, model_features):
    input_df = pd.DataFrame({
        '性别': [input_data['gender']],
        '专业': [input_data['major']],
        '每周学习时长': [input_data['study_hour']],
        '上课出勤率': [input_data['attendance']],
        '期中考试分数': [input_data['mid_score']],
        '作业完成率': [input_data['homework_rate']]
    })
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    for feat in model_features:
        if feat not in input_encoded.columns:
            input_encoded[feat] = 0
    return input_encoded[model_features]

# -------------------------- 5. 页面1：项目介绍（系统预览图用纯文件名） --------------------------
def show_project_intro():
    st.title("学生成绩分析与预测系统")
    st.divider()

    col_overview, col_demo = st.columns([2, 1.2], gap="small")
    with col_overview:
        st.markdown("### 📋 项目概述")
        st.write("本项目是一个基于streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。")

        st.markdown("### ✨ 主要特点")
        st.markdown("""
        - 📊 **数据可视化**：多维度展示学生学业数据  
        - 📈 **专业分析**：按专业分类的详细统计分析  
        - 🤖 **智能预测**：基于机器学习模型的成绩预测  
        - 💡 **学习建议**：根据预测结果提供个性化反馈  
        """)

    with col_demo:
        st.markdown("### 专业数据分析")
        st.markdown("1. 各专业男女性别比例")
        # -------------------------- 核心修改3：系统预览图用纯文件名 --------------------------
        preview_img = "系统预览图.png"  # 仅纯文件名
        if os.path.exists(preview_img):
            st.image(preview_img, use_container_width=True, caption="学生数据分析示意图")
        else:
            st.info(f"（请将'系统预览图.png'放在代码同一目录以显示示意图）")

    st.divider()

    st.markdown("### 🎯 项目目标")
    col_target1, col_target2, col_target3 = st.columns(3, gap="medium")
    with col_target1:
        st.markdown("#### 目标一：分析影响因素")
        st.markdown("""
        - 识别关键学习指标  
        - 探索成绩相关因素  
        - 提供数据支持决策  
        """)
    with col_target2:
        st.markdown("#### 目标二：可视化展示")
        st.markdown("""
        - 专业对比分析  
        - 性别差异研究  
        - 学习模式识别  
        """)
    with col_target3:
        st.markdown("#### 目标三：成绩预测")
        st.markdown("""
        - 机器学习模型  
        - 个性化预测  
        - 及时干预预警  
        """)

    st.divider()

    st.markdown("### 🔧 技术架构")
    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4, gap="small")
    with col_tech1:
        st.markdown("**前端框架**")
        st.markdown("Streamlit")
    with col_tech2:
        st.markdown("**数据处理**")
        st.markdown("Pandas, NumPy")
    with col_tech3:
        st.markdown("**可视化**")
        st.markdown("Plotly, Matplotlib")
    with col_tech4:
        st.markdown("**机器学习**")
        st.markdown("Scikit-learn")

# -------------------------- 6. 页面2：专业数据分析（未修改） --------------------------
def show_major_analysis():
    if student_df is None:
        st.warning("⚠️ 数据未加载，无法进行分析")
        return
    st.title("📊 专业成绩数据分析")
    st.divider()

    st.subheader("1. 各专业男女性别比例")
    col_chart1, col_table1 = st.columns([3, 1], gap="medium")
    with col_chart1:
        gender_ratio = student_df.groupby('专业')['性别'].value_counts(normalize=True).unstack(fill_value=0).round(4)
        if '男' in gender_ratio.columns and '女' in gender_ratio.columns:
            gender_ratio = gender_ratio[['男', '女']]
            gender_ratio.columns = ['男性比例', '女性比例']
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        gender_ratio.plot(kind='bar', stacked=True, ax=ax1, color=['#1f77b4', '#ff7f0e'])
        ax1.set_xlabel("专业")
        ax1.set_ylabel("比例")
        ax1.set_title("各专业男女性别分布")
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig1)
    with col_table1:
        st.write("性别比例数据")
        st.dataframe((gender_ratio * 100).round(2))

    st.subheader("2. 各专业学习指标对比")
    st.caption("(期中/期末成绩 + 每周学习时长)")
    col_chart2, col_table2 = st.columns([3, 1], gap="medium")
    with col_chart2:
        study_metrics = student_df.groupby('专业').agg({
            '期中考试分数': 'mean',
            '期末考试分数': 'mean',
            '每周学习时长': 'mean'
        }).round(4)
        fig2, ax1 = plt.subplots(figsize=(10, 4))
        ax1.set_xlabel('专业', fontsize=10)
        ax1.set_ylabel('分数', color='#1f77b4', fontsize=10)
        line1 = ax1.plot(study_metrics.index, study_metrics['期中考试分数'], marker='o', color='#1f77b4', linewidth=2, label='期中考试分数')
        line2 = ax1.plot(study_metrics.index, study_metrics['期末考试分数'], marker='o', color='#d62728', linewidth=2, label='期末考试分数')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(alpha=0.3, axis='y')
        ax2 = ax1.twinx()
        ax2.set_ylabel('每周学习时长（小时）', color='#2ca02c', fontsize=10)
        line3 = ax2.plot(study_metrics.index, study_metrics['每周学习时长'], marker='o', color='#2ca02c', linewidth=2, label='每周学习时长')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=9)
        ax1.set_title('各专业成绩与学习时长趋势', fontsize=12, pad=15)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)
    with col_table2:
        st.write("学习指标数据")
        st.dataframe(study_metrics, hide_index=False, use_container_width=True)

    st.subheader("3. 各专业出勤率分析")
    col_chart3, col_table3 = st.columns([3, 1], gap="medium")
    with col_chart3:
        attendance_data = student_df.groupby('专业')['上课出勤率'].mean().round(4).to_frame('平均出勤率')
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        attendance_data.plot(kind='bar', ax=ax3, color='#2ca02c')
        ax3.set_xlabel("专业")
        ax3.set_ylabel("平均出勤率")
        ax3.set_title("各专业平均上课出勤率")
        ax3.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig3)
    with col_table3:
        st.write("出勤率数据")
        st.dataframe((attendance_data * 100).round(2))

    st.subheader("4. 大数据管理专业专项分析")
    if '大数据管理' in student_df['专业'].unique():
        bd_df = student_df[student_df['专业'] == '大数据管理']
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric("平均出勤率", f"{bd_df['上课出勤率'].mean()*100:.1f}%")
        with col_kpi2:
            st.metric("平均期末分数", f"{bd_df['期末考试分数'].mean():.1f}分")
        with col_kpi3:
            st.metric("通过率", f"{(bd_df['期末考试分数']>=60).mean()*100:.1f}%")
        with col_kpi4:
            st.metric("平均学习时长", f"{bd_df['每周学习时长'].mean():.1f}小时")
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.write("期末成绩分布")
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.hist(bd_df['期末考试分数'], bins=10, color='#1f77b4')
            ax4.set_xlabel("分数")
            ax4.set_ylabel("人数")
            st.pyplot(fig4)
        with col_dist2:
            st.write("每周学习时长分布")
            fig5, ax5 = plt.subplots(figsize=(5, 4))
            ax5.boxplot(bd_df['每周学习时长'], vert=False)
            ax5.set_xlabel("时长（小时）")
            st.pyplot(fig5)
    else:
        st.info("📌 当前数据集无「大数据管理」专业数据")

# -------------------------- 7. 页面3：成绩预测（等级图片用纯文件名） --------------------------
def show_score_prediction():
    if reg_model is None or model_features is None:
        st.warning("⚠️ 模型未加载，无法预测")
        return
    st.title(f"🎯 {target_col}预测（带等级图片）")
    st.write("📝 输入学生信息，预测分数并匹配优秀/良好/及格/不及格等级图片")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("1. 性别", ["男", "女"])
        major = st.selectbox("2. 专业", unique_majors)
        submit_btn = st.button("🚀 预测成绩", type="primary", use_container_width=True)
    with col2:
        if student_df is not None:
            min_hour = float(student_df['每周学习时长'].min())
            max_hour = float(student_df['每周学习时长'].max())
            default_hour = float(student_df['每周学习时长'].mean())
        else:
            min_hour, max_hour, default_hour = 0.0, 50.0, 15.0
        study_hour = st.slider("3. 每周学习时长（小时）", min_hour, max_hour, default_hour)
        attendance = st.slider("4. 上课出勤率", 0.0, 1.0, 0.9)
        mid_score = st.slider("5. 期中考试分数", 0.0, 100.0, 70.0)
        homework_rate = st.slider("6. 作业完成率", 0.0, 1.0, 0.95)

    if submit_btn:
        try:
            input_data = {
                'gender': gender, 'major': major, 'study_hour': study_hour,
                'attendance': attendance, 'mid_score': mid_score, 'homework_rate': homework_rate
            }
            reg_input = build_reg_input(input_data, model_features)
            pred_score = reg_model.predict(reg_input)[0]
            pred_score = np.clip(pred_score, 0, 100)
            pred_score = round(pred_score, 2)

            # 等级判断（兼容"未及格"图片）
            if pred_score >= 90:
                grade = "优秀"
            elif pred_score >= 70:
                grade = "良好"
            elif pred_score >= 60:
                grade = "及格"
            else:
                grade = "不及格"  # 对应"未及格.PNG"

            # 获取纯文件名图片
            grade_img_path = get_grade_image(grade)

            st.divider()
            st.success("🎉 预测完成！")
            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.metric(f"预测{target_col}", f"{pred_score} 分")
            with col_result2:
                st.metric("成绩等级", grade)

            # 显示图片（纯文件名引用）
            st.subheader("🏆 等级图片匹配")
            if grade_img_path:
                st.image(grade_img_path, width=300, caption=f"{grade}等级（{pred_score}分）")
            else:
                st.info(f"请将{grade}等级图片（{GRADE_IMAGE_MAP[grade][0]}）放在代码同一目录")

            st.subheader("📋 个性化学习建议")
            if grade == "优秀":
                st.success("建议：保持优秀状态，尝试学科竞赛或科研项目，拓展专业能力。")
            elif grade == "良好":
                st.info("建议：针对薄弱知识点加强复习，每周增加1-2小时专项学习。")
            elif grade == "及格":
                st.warning("建议：提高出勤率至90%以上，重点掌握基础知识点，及时请教老师。")
            else:
                st.error("建议：制定紧急学习计划，保证出勤+增加5小时/周学习时间，申请课后辅导。")

        except Exception as e:
            st.error(f"预测出错：{str(e)}")

# -------------------------- 导航菜单 --------------------------
st.sidebar.title("📚 导航菜单")
page = st.sidebar.radio(
    "",
    ["项目介绍", "专业数据分析", "成绩预测"],
    index=0
)

# 页面渲染
if page == "项目介绍":
    show_project_intro()
elif page == "专业数据分析":
    show_major_analysis()
elif page == "成绩预测":
    show_score_prediction()
