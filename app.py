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
risk_categories = [
    "1. Celebrating Suffering (고통 찬양)",
    "2. Violent Acts (폭력 행위)",
    "3. Depicting Violence (폭력적 묘사)",
    "4. Harassment (괴롭힘)",
    "5. Hate Speech (독설)",
    "6. Offensive Language (공격적 언어 표현)",
    "7. Perpetuating Harmful Beliefs (부정적 고정 관념)",
    "8. Adult Content (성인용 콘텐츠)",
    "9. Erotic Content (선정적 콘텐츠)",
    "10. Non-Consensual Nudity (비동의 누드)",
    "11. Monetized Sexual Content (성적 콘텐츠 수익 창출)",
    "12. Endangerment, Harm, or Abuse of Children (아동 유해물)",
    "13. Child Sexual Abuse (아동 성적 학대)",
    "14. Suicidal and Non-suicidal Self-injury (자살, 자해)",
    "15. Political Persuasion (정치적 신조)",
    "16. Influencing Politics (정치 영향)",
    "17. Deterring Democratic Participation (민주주의적 참여 거부)",
    "18. Fraud (도용)",
    "19. Mis/disinformation (허위 정보)",
    "20. Sowing Division (편 가르기)",
    "21. Misrepresentation (정보 왜곡)",
    "22. Types of Defamation (명예훼손)",
    "23. Discriminatory Activities (차별적 표현)",
    "24. Unauthorized Privacy Violations (개인정보 침해)",
    "25. Illegal/Regulated Substances (불법 약물)",
    "26. Illegal Services/Exploitation (불법적 착취)",
    "27. Other Unlawful/Criminal Activities (기타 불법 행위)",
    "28. Increased Inequality and Decline in Employment Quality (불평등 심화 및 고용 질 저하)",
    "29. Economic and Cultural Devaluation of Human Effort (경제적, 문화적 가치 하락)",
    "30. Competitive Dynamics (경쟁적 역학)",
    "31. Overreliance and Unsafe Use (과도한 의존, 비안전한 사용)",
    "32. Loss of Human Agency and Autonomy (인간의 주체성과 자율성 상실)"
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


# 선택된 위험 카테고리 시각화 추가
st.subheader("🔍 위험 카테고리별 상세 분석")

selected_category = st.selectbox("카테고리를 선택하세요", options=risk_categories)

# 선택된 카테고리에 대한 데이터 추출
category_scores = df[selected_category]

# Bar chart 시각화
st.markdown(f"**📈 {selected_category}에 대한 프롬프트별 Risk Score**")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(x=category_scores.index, y=category_scores.values, palette="Reds", ax=ax2)
ax2.set_ylabel("Risk Score")
ax2.set_ylim(0, 5)
st.pyplot(fig2)

# 해당 카테고리 테이블 보기
st.markdown(f"**📋 {selected_category} 점수 테이블**")
category_scores = category_scores.reset_index()
category_scores.columns = ["Prompt Type", "Risk Score"]
st.table(category_scores.style.format({"Risk Score": "{:.2f}"}))
