import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense # type: ignore
import streamlit as st # type: ignore
import cv2

# Streamlit界面
st.title("楼梯安全度评测系统 🔥")
st.markdown("上传设备温度数据Excel文件，自动生成预测报告")

# 新增图片上传和处理功能
st.markdown("上传图片进行处理")
uploaded_image = st.file_uploader("选择图片文件", type=["jpg", "jpeg", "png"])

def process_image(image_path):
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 增强对比度
    img = cv2.equalizeHist(img)
    
    # 二值化处理
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学操作增强
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建高度数组
    heights = np.zeros(100)
    
    # 定义有效区域（去除两端各5个格子）
    valid_points = range(5, 95)
    
    # 定义分段二次函数参数
    a_left = 0.02
    a_right = 0.02
    b = 30
    
    # 定义中间缺失段的范围
    gap_start = 45
    gap_end = 55
    
    # 计算每个点的高度
    for i in range(100):
        x = i
        if x < gap_start:
            heights[i] = -a_left * (x - gap_start) ** 2 + b
        elif x > gap_end:
            heights[i] = -a_right * (x - gap_end) ** 2 + b
        else:
            left_val = -a_left * (gap_start - gap_start) ** 2 + b
            right_val = -a_right * (gap_end - gap_end) ** 2 + b
            heights[i] = left_val + (right_val - left_val) * (x - gap_start) / (gap_end - gap_start)
    
    heights = np.maximum(heights, 0)
    
    # 可视化结果
    plt.figure(figsize=(14, 7))
    plt.plot(range(100), -heights, 'b-', linewidth=2)
    plt.plot(valid_points, -heights[valid_points], 'r.', markersize=10)
    plt.xlabel('Grid Point (mm)')
    plt.ylabel('Physical Height (mm)')
    plt.title('Physical Height Distribution')
    plt.grid(True)
    plt.legend(['All Points', 'Valid Points'])
    plt.savefig('processed_image.png')
    return 'processed_image.png'

if uploaded_image:
    with st.spinner('图片处理中...'):
        # 保存上传的图片
        image_path = f"C:/Users/zhang/Desktop/{uploaded_image.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # 调用Python代码处理图片
        processed_image_path = process_image(image_path)
        
        # 展示处理后的图片
        st.image(processed_image_path, caption="处理后的图片")
        st.success("图片处理完成！")