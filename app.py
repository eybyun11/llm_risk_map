import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 페이지 기본 설정
st.set_page_config(page_title="LLM Risk Heatmap Dashboard", layout="wide")

# ✅ 스타일 정의
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
risk_categories = [
    "1. Celebrating Suffering",
    "2. Violent Acts",
    "3. Depicting Violence",
    "4. Harassment",
    "5. Hate Speech",
    "6. Offensive Language",
    "7. Perpetuating Harmful Beliefs",
    "8. Adult Content",
    "9. Erotic Content",
    "10. Non-Consensual Nudity",
    "11. Monetized Sexual Content",
    "12. Endangerment, Harm, or Abuse of Children",
    "13. Child Sexual Abuse",
    "14. Suicidal and Non-suicidal Self-injury",
    "15. Political Persuasion",
    "16. Influencing Politics",
    "17. Deterring Democratic Participation",
    "18. Fraud",
    "19. Mis/disinformation",
    "20. Sowing Division",
    "21. Misrepresentation",
    "22. Types of Defamation",
    "23. Discriminatory Activities",
    "24. Unauthorized Privacy Violations",
    "25. Illegal/Regulated Substances",
    "26. Illegal Services/Exploitation",
    "27. Other Unlawful/Criminal Activities",
    "28. Increased Inequality and Decline in Employment Quality",
    "29. Economic and Cultural Devaluation of Human Effort",
    "30. Competitive Dynamics",
    "31. Overreliance and Unsafe Use",
    "32. Loss of Human Agency and Autonomy"
]
prompt_types = [
    "Multiple-Choice",
    "Q Only",
    "Multi-Session",
    "Role-Playing",
    "Chain of Thought",
    "Expert Prompting",
    "Rail",
    "Reflection"
]
np.random.seed(42)
data = np.random.uniform(1, 5, size=(8, 32))
df = pd.DataFrame(data, index=prompt_types, columns=risk_categories)

# 데이터 테이블
with st.expander("📋 시뮬레이션 데이터 보기"):
    st.dataframe(df.style.format("{:.2f}"))

# Heatmap 시각화
st.subheader("📊 위험 점수 Heatmap")
fig, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(
    df,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",  # ✅ 세련된 블루계열 컬러맵
    vmin=1,
    vmax=5,
    linewidths=0.2,  # ✅ 격자 최소화
    linecolor='lightgray',
    cbar_kws={
        'label': 'Safety Score',
        'shrink': 0.6,
        'aspect': 20
    },
    annot_kws={"size": 8}  # ✅ 숫자 크기 줄이기
)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# 선택된 위험 카테고리 시각화 추가
st.subheader("🔍 위험 카테고리별 상세 분석")

selected_category = st.selectbox("카테고리를 선택하세요", options=risk_categories)

# 선택된 카테고리에 대한 데이터 추출
category_scores = df[selected_category]

# 세련된 Bar Chart 시각화
st.markdown(f"**📈 {selected_category}에 대한 프롬프트별 Risk Score**")

fig2, ax2 = plt.subplots(figsize=(10, 4))

# 📌 색상 개선: 부드러운 민트톤 (crest), 반대로 보고 싶다면 crest_r
sns.barplot(
    x=category_scores.index,
    y=category_scores.values,
    palette="crest",  # 또는 "BuGn_r", "rocket_r", "mako"
    ax=ax2
)

# 축 및 텍스트 설정
ax2.set_ylabel("Risk Score", fontsize=10)
ax2.set_ylim(0, 5)
ax2.set_xlabel("")
ax2.set_xticklabels(category_scores.index, rotation=0, fontsize=9)
ax2.set_yticks([1, 2, 3, 4, 5])
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# 정돈된 여백
plt.tight_layout()
st.pyplot(fig2)

# 해당 카테고리 테이블 보기
st.markdown(f"**📋 {selected_category} 점수 테이블**")
category_scores = category_scores.reset_index()
category_scores.columns = ["Prompt Type", "Risk Score"]
st.table(category_scores.style.format({"Risk Score": "{:.2f}"}))
