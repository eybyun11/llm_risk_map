import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 페이지 기본 설정
st.set_page_config(page_title="LLM Risk Heatmap Dashboard", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
        .main-title {
            font-size:40px;
            font-weight:bold;
            color:#333333;
            text-align:center;
            margin-bottom: 20px;
        }
        .description {
            font-size:18px;
            color:#666666;
            text-align:center;
            margin-bottom: 30px;
        }
        .footer {
            font-size:14px;
            color:#888888;
            text-align:center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# 상단 타이틀 & 설명
st.markdown('<div class="main-title">🧠 LLM Risk Score Heatmap</div>', unsafe_allow_html=True)
st.markdown('<div class="description">AssureAI의 Risk Score를 시각화합니다.<br>이 대시보드는 모델의 위험 특성 클러스터링과 사전배포 평가 결과를 탐색하는 데 활용됩니다.</div>', unsafe_allow_html=True)

# 데이터 생성
risk_categories = [f"Category_{i+1}" for i in range(32)]
prompt_types = [f"Prompt_{i+1}" for i in range(8)]
np.random.seed(42)
data = np.random.uniform(0, 5, size=(8, 32))
df = pd.DataFrame(data, index=prompt_types, columns=risk_categories)

# 데이터 테이블
with st.expander("📋 시뮬레이션 데이터 보기"):
    st.dataframe(df.style.format("{:.2f}"))

# Heatmap 시각화
st.subheader("📊 위험 점수 Heatmap")
fig, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Risk Score'}, ax=ax)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# 하단 Footer
st.markdown('<div class="footer">ⓒ 2025 AI신뢰성센터 | 이 대시보드는 사전배포 LLM 평가를 위한 시각화 예시입니다.</div>', unsafe_allow_html=True)
