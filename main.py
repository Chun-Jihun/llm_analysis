import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field, conlist
from typing import List, Dict

# --- 0. 환경 설정 및 페이지 구성 ---
st.set_page_config(
    page_title="게임 리뷰 인사이트 대시보드",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# .env 파일에서 API 키 로드
load_dotenv()

# --- 1. LLM 및 Pydantic 모델 정의 ---

# PM 보고서를 위한 구조화된 Pydantic 모델
class PrioritizedIssue(BaseModel):
    issue: str = Field(description="유저가 겪는 핵심 문제점 또는 비판 내용")
    reason: str = Field(description="이 문제가 왜 중요하고 시급한지에 대한 분석 (예: 신규 유저 이탈 유발, 핵심 플레이 경험 저해 등)")
    priority: str = Field(description="문제의 심각성과 빈도를 고려한 우선순위 제안 (예: '매우 높음', '높음', '보통')")

class FeatureOpportunity(BaseModel):
    feature: str = Field(description="유저가 제안하거나 리뷰에서 암시하는 새로운 기능 또는 콘텐츠 아이디어")
    impact: str = Field(description="이 기능이 추가되었을 때 예상되는 긍정적 효과 (예: 리플레이 가치 증대, 유저 만족도 향상 등)")

class PMReport(BaseModel):
    executive_summary: str = Field(description="프로젝트 관리자 및 이해관계자를 위한 핵심 요약. 현재 유저 감성 상태, 가장 중요한 발견, 그리고 즉각적인 조치가 필요한 사항을 요약합니다.")
    key_strengths: List[str] = Field(description="게임의 명확한 강점 및 유저들이 계속 플레이하는 이유. '우리 게임의 USP(Unique Selling Point)'에 해당합니다.")
    prioritized_issues: List[PrioritizedIssue] = Field(description="우선순위가 지정된 핵심 문제점 목록. 개발팀이 즉시 참고할 수 있도록 실행 가능한 형태로 제공합니다.")
    feature_opportunities: List[FeatureOpportunity] = Field(description="유저 피드백 기반의 신규 기능 및 성장 기회 목록.")
    long_term_strategy: List[str] = Field(description="장기적인 게임 로드맵 및 전략적 방향성에 대한 제언.")

@st.cache_resource
def get_llm():
    """LLM 모델을 캐시하여 재로딩을 방지합니다."""
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.stop()
    
    return ChatGoogleGenerativeAI(
        google_api_key=API_KEY,
        model='gemini-1.5-flash',
        temperature=0.7,
    )

def run_llm_analysis(reviews_text: str):
    """Gemini를 사용하여 PM 보고서를 생성합니다."""
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=PMReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 데이터 기반 의사결정을 내리는 숙련된 게임 프로젝트 매니저(PM)입니다. 제공된 유저 리뷰를 심층 분석하여, 비즈니스 목표와 연계된 실행 가능한 인사이트를 담은 보고서를 JSON 형식으로 생성해야 합니다. 단순히 리뷰를 요약하는 것을 넘어, 문제의 우선순위를 정하고, 기회를 포착하며, 장기적인 전략을 제시해주세요."),
        ("human", "아래는 우리 게임에 대한 유저 리뷰 데이터입니다. 이 데이터를 바탕으로 상세한 PM 분석 보고서를 작성해주세요.\n\n--- 리뷰 내용 ---\n{reviews_text}\n\n--- 분석 보고서 ---"),
        ("ai", "{format_instructions}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        return chain.invoke({
            "reviews_text": reviews_text,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
        return None

# --- 2. 데이터 로딩 및 전처리 ---

@st.cache_data
def load_data(file_path):
    """CSV 데이터를 로드하고 기본 전처리를 수행합니다. (캐시 사용)"""
    df = pd.read_csv(file_path)
    df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='s')
    df.rename(columns={'review_text': '리뷰 내용', 'voted_up': '긍정 리뷰'}, inplace=True)
    return df

df_original = load_data('steam_reviews_3430470_korean_limit600_unique.csv')

# --- 3. 사이드바 필터 ---

st.sidebar.header("📊 대시보드 필터")

playtime_range = st.sidebar.slider(
    "플레이 시간(시간) 필터",
    min_value=0,
    max_value=int(df_original['playtime_forever_hours'].max()),
    value=(0, int(df_original['playtime_forever_hours'].quantile(0.95))) # 기본값: 상위 5% 이상치 제외
)

review_type_map = {'모두': None, '긍정적 리뷰 👍': True, '부정적 리뷰 👎': False}
selected_review_type = st.sidebar.selectbox(
    "리뷰 유형 필터",
    options=list(review_type_map.keys())
)

# 필터 적용
df_filtered = df_original[
    (df_original['playtime_forever_hours'] >= playtime_range[0]) &
    (df_original['playtime_forever_hours'] <= playtime_range[1])
]

if review_type_map[selected_review_type] is not None:
    df_filtered = df_filtered[df_filtered['긍정 리뷰'] == review_type_map[selected_review_type]]

# --- 4. 대시보드 UI 구성 ---

st.title("🎮 게임 리뷰 인사이트 대시보드 for PM")
st.markdown(f"**분석 기간:** `{df_original['timestamp_created'].min().date()}` ~ `{df_original['timestamp_created'].max().date()}` | **총 리뷰:** `{len(df_original)}`개 | **필터된 리뷰:** `{len(df_filtered)}`개")


# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 **KPI 요약**",
    "💬 **AI 기반 심층 분석**",
    "⏰ **시계열 분석**",
    "👥 **리뷰어 프로필**",
    "🔍 **원본 데이터 탐색**"
])


with tab1:
    st.header("📈 핵심 성과 지표 (KPI) 요약")
    
    col1, col2, col3 = st.columns(3)
    
    positive_ratio = (df_filtered['긍정 리뷰'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    col1.metric("긍정 리뷰 비율", f"{positive_ratio:.1f}%")
    
    avg_playtime = df_filtered['playtime_forever_hours'].mean()
    col2.metric("평균 플레이 시간 (리뷰 작성 시)", f"{avg_playtime:.1f} 시간")

    avg_votes_up = df_filtered['votes_up'].mean()
    col3.metric("리뷰당 평균 '도움됨' 투표", f"{avg_votes_up:.1f} 개")

    # 긍정/부정 리뷰 비율 파이 차트
    st.subheader("리뷰 유형 분포")
    if not df_filtered.empty:
        fig_pie = px.pie(
            df_filtered, names='긍정 리뷰',
            title='필터된 리뷰의 긍정/부정 비율',
            hole=0.3,
            color_discrete_map={True: 'royalblue', False: 'darkorange'},
            labels={'긍정 리뷰': '리뷰 유형'}
        )
        fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("선택된 필터에 해당하는 리뷰가 없습니다.")

with tab2:
    st.header("💬 AI 기반 심층 분석 (Gemini)")
    st.markdown("필터링된 리뷰 텍스트를 AI 모델에 전달하여 PM 관점의 실행 가능한 인사이트를 도출합니다.")

    if st.button("🤖 필터된 리뷰로 AI 분석 실행하기"):
        if not df_filtered.empty:
            reviews_for_analysis = "\n".join(df_filtered['리뷰 내용'].dropna().head(100)) # 토큰 제한 고려, 100개 샘플
            with st.spinner("Gemini가 리뷰를 분석하고 있습니다. 잠시만 기다려주세요..."):
                report = run_llm_analysis(reviews_for_analysis)
                
                if report:
                    st.subheader("Executive Summary (경영진 요약)")
                    st.info(report['executive_summary'])

                    st.subheader("✅ 강점 (Our Strengths)")
                    for strength in report['key_strengths']:
                        st.success(f"**- {strength}**")
                    
                    st.subheader("🔥 개선 우선순위 (Issues to Prioritize)")
                    for issue in report['prioritized_issues']:
                        with st.expander(f"**{issue['priority']}**: {issue['issue']}"):
                            st.write(f"**분석:** {issue['reason']}")
                    
                    st.subheader("🚀 신규 기회 (Feature Opportunities)")
                    for opp in report['feature_opportunities']:
                        with st.container(border=True):
                            st.markdown(f"**기능/콘텐츠:** {opp['feature']}")
                            st.markdown(f"**기대 효과:** {opp['impact']}")
                            
                    st.subheader("🗺️ 장기 전략 제안 (Long-term Strategy)")
                    for strategy in report['long_term_strategy']:
                        st.markdown(f"- {strategy}")
                else:
                    st.error("분석 보고서를 생성하지 못했습니다.")
        else:
            st.warning("분석할 리뷰가 없습니다. 필터를 조정해주세요.")

with tab3:
    st.header("⏰ 리뷰 감성 시계열 분석")
    st.markdown("시간의 흐름에 따른 긍정/부정 리뷰의 추세를 확인하여 특정 업데이트나 이벤트의 영향을 파악할 수 있습니다.")
    
    # 주(Week) 단위로 데이터 리샘플링
    df_resampled = df_filtered.set_index('timestamp_created').resample('W').agg(
        positive_reviews=('긍정 리뷰', lambda x: (x == True).sum()),
        negative_reviews=('긍정 리뷰', lambda x: (x == False).sum())
    ).reset_index()

    fig_timeline = px.bar(
        df_resampled,
        x='timestamp_created',
        y=['positive_reviews', 'negative_reviews'],
        title='주별 긍정/부정 리뷰 수 추이',
        labels={'timestamp_created': '날짜', 'value': '리뷰 수', 'variable': '리뷰 유형'},
        color_discrete_map={'positive_reviews': 'royalblue', 'negative_reviews': 'darkorange'}
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with tab4:
    st.header("👥 리뷰어 프로필 분석")
    st.markdown("어떤 성향의 유저들이 우리 게임에 대해 리뷰를 남기는지 파악하여 핵심 타겟 고객을 이해합니다.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("유저의 게임 보유 수 분포")
        fig_games_owned = px.histogram(
            df_filtered, x='num_games_owned', nbins=50,
            title="리뷰어의 스팀 라이브러리 규모",
            labels={'num_games_owned': '보유 게임 수'}
        )
        st.plotly_chart(fig_games_owned, use_container_width=True)

    with col2:
        st.subheader("유저의 리뷰 작성 수 분포")
        fig_num_reviews = px.histogram(
            df_filtered, x='num_reviews_by_author', nbins=50,
            title="리뷰어의 스팀 리뷰 활동성",
            labels={'num_reviews_by_author': '작성한 리뷰 수'}
        )
        st.plotly_chart(fig_num_reviews, use_container_width=True)
        
    st.markdown("---")
    st.subheader("구매 vs 무료 제공 유저 비교")
    purchase_comparison = df_filtered.groupby('steam_purchase')['긍정 리뷰'].value_counts(normalize=True).unstack().fillna(0) * 100
    st.dataframe(purchase_comparison.style.format("{:.1f}%"))
    st.caption("`True`는 스팀 상점 구매, `False`는 키 등록 등 다른 경로로 획득한 경우입니다.")


with tab5:
    st.header("🔍 원본 데이터 탐색")
    st.markdown("필터링된 리뷰 원본 데이터를 직접 확인하고 정렬하거나 검색할 수 있습니다.")
    
    st.dataframe(df_filtered[[
        '리뷰 내용', '긍정 리뷰', 'playtime_forever_hours', 'votes_up', 'timestamp_created', 'steam_purchase'
    ]], use_container_width=True, height=500)