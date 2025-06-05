import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import uuid

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
st.markdown('<div class="main-title">🧠 LLM Safety Score Heatmap</div>', unsafe_allow_html=True)
st.markdown('<div class="description">AssureAI의 Safety Score를 시각화합니다.<br>이 대시보드는 모델의 위험 특성 클러스터링과 사전배포 평가 결과를 탐색하는 데 활용됩니다.</div>', unsafe_allow_html=True)

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


# 데이터 테이블
with st.expander("📋 시뮬레이션 데이터 보기"):
    st.dataframe(df.style.format("{:.2f}"))


st.markdown("---")

# 선택된 위험 카테고리 시각화 추가
st.subheader("🔍 위험 카테고리별 상세 분석")

selected_category = st.selectbox("카테고리를 선택하세요", options=risk_categories)

# 선택된 카테고리에 대한 데이터 추출
category_scores = df[selected_category]

# 세련된 Bar Chart 시각화
st.markdown(f"**📈 {selected_category}에 대한 프롬프트별 Safety Score**")

fig2, ax2 = plt.subplots(figsize=(10, 4))

# ✅ 색상 매핑: YlGnBu 컬러맵 적용
cmap = cm.get_cmap('YlGnBu')
norm = plt.Normalize(1, 5)
colors = cmap(norm(category_scores.values))

sns.barplot(
    x=category_scores.index,
    y=category_scores.values,
    palette=colors,  # 점수 기반 색상 지정
    ax=ax2
)

# ✅ 축 및 폰트 설정
ax2.set_ylabel("Score", fontsize=10)
ax2.set_ylim(0, 5)
ax2.set_xlabel("")
ax2.set_xticklabels(category_scores.index, rotation=0, fontsize=9)
ax2.set_yticks([1, 2, 3, 4, 5])
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.title(f"{selected_category} – Prompt Type Scores", fontsize=14, pad=10)
plt.tight_layout()
st.pyplot(fig2)


# 데이터 테이블
with st.expander(f"📋 {selected_category} 점수 테이블"):
    category_scores = category_scores.reset_index()
    category_scores.columns = ["Prompt Type", "Safety Score"]
    st.table(category_scores.style.format({"Safety Score": "{:.2f}"}))
    
# 샘플 대화 생성 함수
def generate_sample_dialogue():
    return [
        {"user": "이건 무엇인가요?"},
        {"model": "이건 예시 응답입니다. 실제로는 모델이 여기에 답변을 생성합니다."},
        {"user": "좀 더 자세히 알려줄 수 있어?"},
        {"model": "물론이죠. 이 부분에 대해 더 자세히 설명드리겠습니다."}
    ]

# 데이터 생성
chat_dataset = {}
for category in risk_categories:
    for prompt in prompt_types:
        key = (category, prompt)
        chat_dataset[key] = []
        for i in range(random.randint(1, 3)):  # 각 조합에 대해 1~3개 샘플
            dialogue_id = str(uuid.uuid4())[:8]
            dialogue = generate_sample_dialogue()
            chat_dataset[key].append((dialogue_id, dialogue))

# -------------------------------
# 스타일 정의 (고급 말풍선)
# -------------------------------
st.markdown("""
    <style>
    .chat-container {
        margin-bottom: 28px;
        width: 100%;
    }
    .bubble {
        max-width: 80%;
        padding: 14px 20px;
        border-radius: 16px;
        position: relative;
        font-size: 15.2px;
        line-height: 1.6;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
        color: #1e1e1e;
        font-family: 'Segoe UI', sans-serif;
    }

    .user {
        background: #dce3f0;
        margin-left: auto;
        margin-right: 12px;
        text-align: right;
    }

    .user:after {
        content: "";
        position: absolute;
        right: -10px;
        top: 12px;
        border-top: 10px solid transparent;
        border-left: 10px solid #dce3f0;
        border-bottom: 10px solid transparent;
    }

    .model {
        background: #d3f3e2;
        margin-right: auto;
        margin-left: 12px;
        text-align: left;
    }

    .model:before {
        content: "";
        position: absolute;
        left: -10px;
        top: 12px;
        border-top: 10px solid transparent;
        border-right: 10px solid #d3f3e2;
        border-bottom: 10px solid transparent;
    }

    .label {
        font-weight: 500;
        font-size: 15px;
        margin-bottom: 6px;
        font-family: 'Segoe UI', sans-serif;
    }

    .user-label {
        text-align: right;
        margin-right: 12px;
        color: #4a5568;
    }

    .model-label {
        text-align: left;
        margin-left: 12px;
        color: #2f855a;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.subheader("📊 대화 조회")

selected_category = st.selectbox("📂Select Risk Category", risk_categories)
selected_prompt = st.selectbox("🧠Select Prompt Type", prompt_types)

filtered_chats = chat_dataset.get((selected_category, selected_prompt), [])

if not filtered_chats:
    st.warning("해당 조합에 대한 대화 샘플이 없습니다.")
else:
    st.markdown(f"### 🔍 조회 결과: {len(filtered_chats)}건")
    id_list = [f"{i+1}. Dialogue ID: {d_id}" for i, (d_id, _) in enumerate(filtered_chats)]
    selected_label = st.selectbox("🗂️ 대화 ID 선택", id_list)

    selected_index = id_list.index(selected_label)
    selected_dialogue = filtered_chats[selected_index][1]

    st.markdown("---")
    st.markdown("### 💬 대화 내용")

    for turn in selected_dialogue:
        if "user" in turn:
            st.markdown(f"""
            <div class="chat-container">
                <div class="label user-label">👤 사용자</div>
                <div class="bubble user">{turn['user']}</div>
            </div>
            """, unsafe_allow_html=True)
        elif "model" in turn:
            st.markdown(f"""
            <div class="chat-container">
                <div class="label model-label">🤖 모델</div>
                <div class="bubble model">{turn['model']}</div>
            </div>
            """, unsafe_allow_html=True)
