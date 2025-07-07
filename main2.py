import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict
from fpdf import FPDF

# --- 0. 환경 설정 및 페이지 구성 ---
st.set_page_config(
    page_title="인공지능 게임 분석",
    layout="wide",
    initial_sidebar_state="expanded"
)

# .env 파일에서 API 키 로드
load_dotenv()

# --- 1. 다중 전문가 LLM 및 Pydantic 모델 정의 ---

# (KeywordCluster, QualitativeReport, StrategicPriority, StrengthAnalysis, UserSegment, SegmentationReport 모델은 이전과 동일)
class KeywordCluster(BaseModel):
    theme: str = Field(description="리뷰에서 공통적으로 나타나는 주제 또는 테마 (예: 'UI/UX 불편함', '전투 밸런스 문제', '풍부한 스토리')")
    keywords: List[str] = Field(description="해당 테마를 대표하는 핵심 키워드 목록")
    sentiment: str = Field(description="해당 테마에 대한 전반적인 감성 (예: '긍정적', '부정적', '복합적')")
    review_examples: List[str] = Field(description="해당 테마를 잘 보여주는 대표적인 유저 리뷰 2~3개 요약")

class QualitativeReport(BaseModel):
    keyword_clusters: List[KeywordCluster] = Field(description="주요 키워드 클러스터링 및 감성 분석 결과")
    emerging_issues: List[str] = Field(description="최근들어 빈도가 증가하거나 새롭게 나타나는 불만 키워드 또는 주제")

class StrategicPriority(BaseModel):
    issue: str = Field(description="개선이 필요한 핵심 문제점")
    impact_analysis: str = Field(description="이 문제가 유저 경험과 비즈니스에 미치는 영향 분석")
    recommendation: str = Field(description="문제 해결을 위한 구체적인 액션 아이템 제안")
    expected_roi: str = Field(description="개선 시 예상되는 ROI (Return on Investment) 또는 긍정적 효과 (예: '높음', '중간', '낮음')")

class StrengthAnalysis(BaseModel):
    core_strengths: List[str] = Field(description="유저 충성도를 유지시키는 현재 게임의 핵심 강점 목록")
    strategic_priorities: List[StrategicPriority] = Field(description="개선 시 ROI가 높을 것으로 예상되는 문제점 및 해결 방안 목록")

class UserSegment(BaseModel):
    segment_name: str = Field(description="플레이 스타일, 과금 패턴 등을 기반으로 정의된 유저 군집의 이름 (예: '하드코어 플레이어', '스토리 탐험가', '무과금 유저')")
    characteristics: str = Field(description="해당 세그먼트의 주요 특징 요약")
    feedback_summary: str = Field(description="해당 세그먼트에서 주로 나타나는 긍정/부정 피드백 요약")

class SegmentationReport(BaseModel):
    user_segments: List[UserSegment] = Field(description="리뷰 내용 기반으로 추론한 주요 유저 세그먼트별 분석")

class FutureProposal(BaseModel):
    timeframe: str = Field(description="제안의 시간적 범위 (단기, 중기, 장기)")
    proposal_type: str = Field(description="제안의 종류 (콘텐츠, 시스템 개선, BM, 전략 등)")
    description: str = Field(description="제안에 대한 구체적인 내용")
    rationale: str = Field(description="이 제안이 필요한 이유 및 기대효과")
    
# 1.1. 신규 기능/콘텐츠 제안을 위한 Pydantic 모델
class SuggestedFeature(BaseModel):
    feature_name: str = Field(description="제안하는 신규 기능 또는 콘텐츠의 이름")
    description: str = Field(description="어떤 기능이며, 유저의 어떤 문제나 요구를 해결해주는지에 대한 구체적인 설명")
    expected_impact: str = Field(description="도입 시 기대되는 긍정적 효과 (예: '신규 유저 안착률 증가', '기존 유저의 장기 플레이 동기 부여')")

class FutureStrategyReport(BaseModel):
    """향후 콘텐츠 및 전략 제안 전문가의 보고서 (수정)"""
    proposals: List[FutureProposal] = Field(description="게임의 미래를 위한 단기/중기/장기적 제안 목록")
    suggested_new_features: List[SuggestedFeature] = Field(description="리뷰에서 암시된 유저의 unmet needs를 기반으로, 도입을 고려해볼 만한 구체적인 신규 기능 또는 콘텐츠 목록")

# 1.2. 소수 의견 분석을 위한 신규 Pydantic 모델
class MinorityOpinion(BaseModel):
    opinion_summary: str = Field(description="소수 의견의 핵심 내용 요약")
    potential_insight: str = Field(description="이 소수 의견이 왜 중요하며, 어떤 잠재적 가치(예: 새로운 성장 기회, 심각한 잠재적 리스크)를 담고 있는지에 대한 분석")
    reviewer_quote: str = Field(description="해당 의견을 가장 잘 보여주는 리뷰 원문 인용 또는 요약")

class MinorityReport(BaseModel):
    """소수 의견 분석 전문가의 보고서"""
    minority_opinions: List[MinorityOpinion] = Field(description="다수의 의견에 묻혔지만 중요한 인사이트를 담고 있는 소수 의견 목록")

# 1.3. 최종 보고서를 위한 Pydantic 모델
class FinalReport(BaseModel):
    """인공지능 분석가의 최종 종합 보고서"""
    executive_summary: str = Field(description="경영진 및 핵심 의사결정자를 위한 보고서의 핵심 요약. 현재 상황, 가장 시급한 과제, 그리고 가장 큰 기회를 요약합니다.")
    top_3_strengths: List[str] = Field(description="모든 분석을 종합했을 때, 우리 게임이 가진 가장 중요한 강점 3가지")
    top_3_priorities: List[str] = Field(description="다음 분기에 가장 먼저 해결해야 할 가장 중요한 과제 3가지")
    decision_points: List[str] = Field(description="경영진이나 PM이 시급히 결정을 내려야 할 주요 안건 목록 (예: '신규 콘텐츠 개발과 시스템 안정화 중 리소스 배분 결정')")



# 1.3. LLM 호출 및 캐싱
@st.cache_resource
def get_llm():
    """LLM 모델을 캐시하여 재로딩을 방지합니다."""
    # Streamlit Community Cloud의 Secrets에서 API 키를 가져오도록 수정
    # 로컬 테스트 시에는 기존 .env 방식도 작동하도록 or 연산자 사용
    API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    
    if not API_KEY:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일 또는 Streamlit Secrets를 확인해주세요.")
        st.stop()
    
    return ChatGoogleGenerativeAI(
        google_api_key=API_KEY,
        model='gemini-2.0-flash',
        temperature=0.7,
        model_kwargs={"response_mime_type": "application/json"} 
    )

# 1.4. 전문가별 프롬프트 템플릿 정의
qualitative_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """당신은 숙련된 텍스트 마이닝 전문가이자 게임 UX 분석가입니다. 
당신의 임무는 유저 리뷰 텍스트를 분석하여 핵심 주제(theme)와 감정(sentiment)을 분류하고, 유의미한 클러스터를 도출하는 것입니다.

**분석 목적:** 이 분석은 PM과 기획자가 유저 불만, 개선 요구, 칭찬 포인트를 이해하고, 다음 업데이트 방향을 정하는 데 도움이 되도록 작성되어야 합니다.

**분석 기준:**
- `theme`: 단순 키워드가 아닌, 유저가 경험한 기능 또는 맥락 기반으로 테마를 정의해야 합니다. (예: 단순 '버그' → '전투 중 발생하는 클리어 불가능 버그')
- `sentiment`: 반드시 유저 감정의 뉘앙스를 파악해서 '긍정적', '부정적', '복합적' 중 하나로 정확하게 분류해야 합니다.
- `review_examples`: 감정과 theme의 성격이 명확하게 드러나는 리뷰 요약으로 구성하세요. 의역 없이 최대한 실제 표현을 보존하세요.

**추가 기준:**
- `emerging_issues`: 최근에 등장하거나, 증가 추세인 신종 불만을 감지해 목록화하세요. 이전 리뷰와 구별되는 패턴이 있어야 합니다.

반드시 아래 형식 지침에 맞춰 JSON으로 출력하세요:
{format_instructions}"""
    ),
    ("human",
     """다음은 유저 리뷰 원문입니다. 이 데이터를 분석하여 위 기준에 따라 정성 리뷰 분석 보고서를 작성하세요.

--- 유저 리뷰 ---
{reviews_text}"""
    )
])

strategic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """당신은 ROI 중심의 전략 분석가이자 게임 밸런스 컨설턴트입니다. 
유저 리뷰를 통해 현재 게임의 강점 및 개선 과제를 분석하고, 비즈니스 영향과 ROI 기준으로 정리해야 합니다.

**분석 목적:** 이 분석은 경영진이 다음 분기의 업데이트 우선순위를 정할 때 참고할 전략 문서입니다.

**분석 기준:**
- `core_strengths`: 유저의 긍정적인 반응이 반복적으로 나타나는 요소를 중심으로 도출하세요. 반드시 리뷰 내 근거가 있어야 하며, "그럴듯해 보이는 일반론"은 제외합니다.
- `strategic_priorities`: 유저 불만 중, 수정이 실제 개선 효과(ROI)가 클 것으로 예상되는 문제만 정리하세요.
    - `issue`: 문제가 되는 기능 또는 경험을 명확히 기술하세요.
    - `impact_analysis`: 유저 이탈, 충성도 저하, 수익 저하 등 구체적인 영향을 기술하세요.
    - `recommendation`: 어떤 방식으로 개선할 수 있는지, 현실적인 액션 아이템을 제시하세요.
    - `expected_roi`: 수정 시 기대되는 성과를 직관적으로 평가하세요. ("높음", "중간", "낮음")

**유의사항:**
- '많이 언급되었으나 수정해도 효과가 적은 이슈'는 제외하세요.
- 모든 항목은 리뷰에 기반해야 하며, 개인 추정이나 감성적 언급은 배제하세요.

아래 출력 형식 지침을 반드시 따르세요:
{format_instructions}"""
    ),
    ("human",
     """다음은 유저 리뷰 원문입니다. 이 데이터를 분석하여 위 기준에 따라 강점 및 개선 우선순위 보고서를 작성하세요.

--- 유저 리뷰 ---
{reviews_text}"""
    )
])

segmentation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """당신은 유저 리서치 전문가이며, 게임 유저의 심리와 행태를 기반으로 세그먼트를 정의하고 피드백을 분석합니다.

**분석 목적:** 기획자와 마케팅 팀이 타겟 유저에 맞는 콘텐츠를 기획하고, 이탈 방지 전략을 세울 수 있도록 돕기 위한 보고서입니다.

**분석 방식:**
- `segment_name`: 유저가 보여주는 리뷰 스타일, 과금 패턴, 플레이 목적 등을 기반으로 직관적인 명칭을 붙이세요. (예: '하드코어 파밍 유저', '스토리 중심 탐험가')
- `characteristics`: 리뷰 내 표현을 바탕으로 각 세그먼트의 특징을 요약하세요. 단순 수치가 아닌 '의도', '불만', '칭찬'에 초점을 맞추세요.
- `feedback_summary`: 각 세그먼트가 가진 주요 감정/이슈/칭찬 내용을 요약합니다.

아래 형식을 반드시 따르세요:
{format_instructions}"""
    ),
    ("human",
     """다음은 유저 리뷰 원문입니다. 이 데이터를 분석하여 위 기준에 따라 유저 세그먼트 분석 보고서를 작성하세요.

--- 유저 리뷰 ---
{reviews_text}"""
    )
])

future_strategy_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """당신은 크리에이티브 디렉터이자 수석 게임 플래너입니다. 
유저 피드백을 바탕으로 미래 콘텐츠, 시스템, 전략 방향성을 제시하는 역할입니다.

**분석 목적:** 다음 시즌 또는 업데이트 기획에 반영할 수 있는 전략 아이디어를 제시하세요.

**분석 기준:**
- `timeframe`: 반드시 제안이 단기/중기/장기 중 어디에 해당하는지 명시하세요.
- `proposals`: 단기/중기/장기적 관점에서 게임이 나아가야 할 방향성을 제시하세요.
- `proposal_type`: 콘텐츠, 시스템 개선, 비즈니스 모델(BM), 운영 전략 등 구체적인 분류
- `description`: 실현 가능한 수준의 구체적인 아이디어 제시
- `rationale`: 유저 리뷰에서 왜 이 제안이 도출되었는지 반드시 설명하고, 기대효과를 기술하세요.
- `suggested_new_features`: 리뷰 내용에 암시된 유저의 숨겨진 니즈(Unmet Needs)를 포착하여, 즉시 도입을 검토해볼 만한 **구체적인 신규 기능 또는 콘텐츠**를 제안하세요. '무엇을', '왜', '어떤 효과를 기대하는지' 명확히 기술해야 합니다. 추상적인 방향성 제시는 지양합니다.

반드시 아래 형식을 따르세요:
{format_instructions}"""
    ),
    ("human",
     """다음은 유저 리뷰 원문입니다. 이 데이터를 분석하여 위 기준에 따라 향후 전략 및 신규 기능/콘텐츠 제안 보고서를 작성하세요.

--- 유저 리뷰 ---
{reviews_text}"""
    )
])

# 2.2. 소수 의견 분석가 프롬프트 신규 추가
minority_opinion_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """당신은 예리한 통찰력을 가진 데이터 분석가로, '숨은 보석 찾기'의 전문가입니다.
당신의 임무는 다수의 목소리에 묻히기 쉬운 소수의견 중에서, 잠재적 가치가 매우 높은 의견을 발굴하는 것입니다.

**분석 목적:** 주류 의견에서는 놓치고 있는 새로운 성장 기회나 심각한 잠재적 리스크를 조기에 발견하여, 남들보다 한발 앞선 의사결정을 지원합니다.

**분석 기준:**
- **독창성:** 일반적인 칭찬이나 불만(예: "재미있어요", "버그 많아요")은 철저히 무시하세요.
- **논리성:** 소수 의견이지만, 그 주장에 대한 논리적인 근거나 설득력 있는 이유를 제시하는 리뷰를 찾아야 합니다.
- **잠재력:** 해당 의견이 만약 제품에 반영되었을 때, 큰 파급효과(예: 새로운 유저층 유입, 코어 팬덤 강화, 심각한 문제 사전 예방)를 가져올 수 있는 것이어야 합니다.

반드시 아래 형식 지침에 맞춰 JSON으로 출력하세요:
{format_instructions}"""),
    ("human", "다음 유저 리뷰에서 다수가 놓치고 있는 핵심적인 소수 의견을 찾아 분석 보고서를 작성하세요.\n\n--- 유저 리뷰 ---\n{reviews_text}")
])

# 1.5. LLM 호출 함수
def run_expert_analysis(reviews_text: str, expert_prompt: ChatPromptTemplate, pydantic_model):
    """개별 전문가 LLM을 실행하는 범용 함수"""
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=pydantic_model)
    
    chain = expert_prompt | llm | parser
    
    try:
        return chain.invoke({
            "reviews_text": reviews_text,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        expert_role = expert_prompt.messages[0].prompt.template.split('\n')[0]
        st.error(f"'{expert_role}' 역할 분석 중 오류 발생: {e}")
        return None

def run_final_synthesis(expert_reports: Dict):
    """모든 전문가 보고서를 종합하여 최종 결론을 도출하는 함수"""
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=FinalReport)

    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 게임 회사의 수석 프로젝트 매니저(Head of Product)입니다. 당신의 팀에 소속된 각 분야별 전문 분석가들이 제출한 보고서를 받았습니다. 이 보고서들을 종합적으로 검토하여, 경영진이 빠르고 정확한 의사결정을 내릴 수 있도록 최종 보고서를 작성해주세요. 각 분석가의 편향된 시각을 넘어 전체적인 비즈니스 관점에서 핵심만 요약해야 합니다. 반드시 다음 출력 형식 지침을 준수해주세요.\n{format_instructions}"),
        ("human", "아래는 전문가들의 분석 보고서입니다. 이를 바탕으로 최종 종합 보고서를 작성해주세요.\n\n--- 전문가 보고서 ---\n{reports_text}")
    ])

    chain = synthesis_prompt | llm | parser
    
    try:
        # invoke 시점에 format_instructions 값을 함께 전달합니다.
        return chain.invoke({
            "reports_text": str(expert_reports),
            "format_instructions": parser.get_format_instructions()
        })
    # --- END: 수정된 부분 ---
    except Exception as e:
        st.error(f"최종 보고서 종합 중 오류 발생: {e}")
        return None

# --- 3. PDF 생성 함수 ---

class PDF(FPDF):
    def header(self):
        # 폰트 파일이 프로젝트 폴더에 있어야 합니다.
        self.add_font('NanumGothic', '', 'NanumGothic.ttf', uni=True)
        self.set_font('NanumGothic', '', 15)
        self.cell(0, 10, '인공지능 게임 분석 종합 보고서', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('NanumGothic', '', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('NanumGothic', '', 10)
        self.multi_cell(0, 6, body)
        self.ln()

    def add_report_section(self, title, content_dict):
        self.add_page()
        self.chapter_title(title)
        for key, value in content_dict.items():
            self.set_font('NanumGothic', '', 11)
            self.cell(0, 8, f"■ {key}", 0, 1, 'L')
            self.chapter_body(str(value))

def create_pdf_report(report_data):
    pdf = PDF()
    pdf.add_font('NanumGothic', '', 'NanumGothic.ttf', uni=True)
    
    # 최종 결론 요약
    final_report = report_data.get('final')
    if final_report:
        pdf.add_page()
        pdf.chapter_title("최종 결론 (Executive Summary)")
        pdf.chapter_body(final_report['executive_summary'])
        
        pdf.chapter_title("Top 3 강점")
        for item in final_report['top_3_strengths']:
            pdf.chapter_body(f"- {item}")
            
        pdf.chapter_title("Top 3 개선 과제")
        for item in final_report['top_3_priorities']:
            pdf.chapter_body(f"- {item}")
            
        pdf.chapter_title("주요 결정 필요 사안")
        for item in final_report['decision_points']:
            pdf.chapter_body(f"- {item}")

    # (필요 시 다른 전문가 리포트도 PDF에 추가 가능)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 2. 데이터 로딩 및 전처리 ---
# --- 4. 데이터 로딩 및 필터링 로직 ---

@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='s')
    df.rename(columns={'review_text': '리뷰 내용', 'voted_up': '긍정 리뷰'}, inplace=True)

    # --- 업데이트 버전 데이터 시뮬레이션 ---
    update_dates = {
        "v1.0 (출시)": "2023-01-01", "v1.1 (편의성 개선)": "2024-03-15",
        "v1.2 (신규 캐릭터 추가)": "2024-05-20", "v1.3 (밸런스 패치)": "2024-06-30",
    }
    update_datetimes = {v: pd.to_datetime(d) for v, d in update_dates.items()}
    def get_version(ts):
        for v, d in reversed(list(update_datetimes.items())):
            if ts >= d: return v
        return "알 수 없음"
    df['version'] = df['timestamp_created'].apply(get_version)
    
    # --- 플레이 시간 및 리뷰 수 구간(Bin) 생성 (수정된 로직) ---
    max_playtime = int(df['playtime_forever_hours'].max())
    
    # [수정 1] 플레이 시간 구간 생성 로직을 더 명확하게 변경
    if max_playtime > 0:
        # 1. 구간의 경계를 명시적으로 생성
        bin_edges = list(range(0, max_playtime + 6, 5))
        # 2. 생성된 경계에 맞춰 정확한 개수의 라벨을 생성 (라벨 길이 = 경계 길이 - 1)
        labels = [f'{bin_edges[i]} - {bin_edges[i+1] - 1}시간' for i in range(len(bin_edges) - 1)]
        df['playtime_bin'] = pd.cut(df['playtime_forever_hours'], bins=bin_edges, right=False, labels=labels)
    else:
        df['playtime_bin'] = '0-4시간' # 플레이 시간이 0인 경우 대비

    # [수정 2] 리뷰 작성 수 구간 생성 시 right=True 옵션으로 논리 오류 수정
    review_count_bins = [0, 1, 5, 10, 50, float('inf')]
    review_count_labels = ['첫 리뷰어 (1개)', '가끔 작성 (2-5개)', '나름 활발 (6-10개)', '리뷰 전문가 (11-50개)', '매우 전문적 (51개 이상)']
    # right=True는 (0, 1], (1, 5] 와 같은 구간을 의미하며, 이 경우가 의도에 맞음
    df['review_count_bin'] = pd.cut(df['num_reviews_by_author'], bins=review_count_bins, right=True, labels=review_count_labels)
    
    return df

# 데이터 로딩 함수 호출 (이 부분은 기존과 동일)
df_original = load_and_preprocess_data('steam_reviews_3430470_korean_limit600_unique.csv')


# --- 5. 사이드바 UI (필터 전면 개편) ---
st.sidebar.header("상세 필터")

# 5.1. 플레이 시간 구간 필터
playtime_options = ['모두'] + df_original['playtime_bin'].cat.categories.tolist()
selected_playtime = st.sidebar.selectbox("플레이 시간 구간", options=playtime_options)

# 5.2. 업데이트 버전 필터
version_options = ['모두'] + df_original['version'].unique().tolist()
selected_version = st.sidebar.selectbox("리뷰 작성 시점 (버전)", options=version_options)

# 5.3. 유저 리뷰 수 필터
review_count_options = ['모두'] + df_original['review_count_bin'].cat.categories.tolist()
selected_review_count = st.sidebar.selectbox("유저의 리뷰 활동성", options=review_count_options)


# --- 필터 적용 로직 ---
df_filtered = df_original.copy()
if selected_playtime != '모두':
    df_filtered = df_filtered[df_filtered['playtime_bin'] == selected_playtime]
if selected_version != '모두':
    df_filtered = df_filtered[df_filtered['version'] == selected_version]
if selected_review_count != '모두':
    df_filtered = df_filtered[df_filtered['review_count_bin'] == selected_review_count]


# --- 4. 대시보드 UI 구성 ---

# 수정: title에서 이모지 제거
st.title("인공지능 게임 분석 대시보드")
st.markdown(f"**분석 기간:** `{df_original['timestamp_created'].min().date()}` ~ `{df_original['timestamp_created'].max().date()}` | **총 리뷰:** `{len(df_original)}`개 | **필터된 리뷰:** `{len(df_filtered)}`개")

# 수정: tabs에서 이모지 및 마크다운 제거
tab1, tab2, tab3, tab4 = st.tabs([
    "AI 전문가 종합 분석", 
    "데이터 기반 현황", 
    "소수 의견 심층 분석", 
    "원본 데이터 탐색"
    ])


with tab1:
    # 수정: header에서 이모지 제거
    st.header("AI 종합 분석 (Multi-Agent AI Analysis)")
    st.markdown("AI가 리뷰를 분석하고, 인사이트를 도출합니다.")

    # 수정: button에서 이모지 제거
    if st.button("AI 분석 실행하기", key="main_analysis_button"):
        if not df_filtered.empty:
            reviews_for_analysis = "\n".join(df_filtered.sort_values('timestamp_created', ascending=False)['리뷰 내용'].dropna().head(150))
            st.session_state['reviews_for_analysis'] = reviews_for_analysis

            
            with st.spinner("AI가 분석을 시작합니다... (최대 1-2분 소요)"):
                expert_reports = {}
                
                with st.status("1/4: 텍스트 마이닝 및 UX 분석 중...", expanded=True) as status:
                    expert_reports['qualitative'] = run_expert_analysis(
                        reviews_for_analysis, qualitative_prompt, QualitativeReport
                    )
                    
                    st.write("정성 리뷰 분석 완료.")
                    status.update(label="텍스트 마이닝/UX 분석 완료!", state="complete", expanded=False)

                with st.status("2/4: ROI 기반 전략 분석 중...", expanded=True) as status:
                    expert_reports['strategic'] = run_expert_analysis(
                        reviews_for_analysis, strategic_prompt, StrengthAnalysis
                    )
                    st.write("전략 분석 완료.")
                    status.update(label="ROI 전략 분석 완료!", state="complete", expanded=False)

                with st.status("3/4: 유저 세그먼트 분석 중...", expanded=True) as status:
                    expert_reports['segmentation'] = run_expert_analysis(
                        reviews_for_analysis, segmentation_prompt, SegmentationReport
                    )
                    st.write("유저 세그먼트 분석 완료.")
                    status.update(label="유저 세그먼트 분석 완료!", state="complete", expanded=False)
                    
                with st.status("4/4: 미래 전략 및 콘텐츠 제안 중...", expanded=True) as status:
                    expert_reports['future'] = run_expert_analysis(
                        reviews_for_analysis, future_strategy_prompt, FutureStrategyReport
                    )
                    st.write("미래 전략 제안 완료.")
                    status.update(label="미래 전략/콘텐츠 제안 완료!", state="complete", expanded=False)

                if all(report is not None for report in expert_reports.values()):
                    with st.spinner("최종 보고서 종합 중..."):
                        final_report = run_final_synthesis(expert_reports)
                        st.session_state['analysis_report'] = {**expert_reports, 'final': final_report}
                        st.session_state['analysis_complete'] = True
            st.rerun()
        else:
            st.warning("분석할 리뷰가 없습니다. 필터를 조정해주세요.")
    if st.session_state.get('analysis_complete', False):
        report_data = st.session_state.get('analysis_report', {})
        final_report = report_data.get('final')
        if final_report:
            # 수정: subheader에서 이모지 제거
            st.subheader("최종 결론 (Executive Summary)")
            st.info(f"**핵심 요약:**\n\n{final_report['executive_summary']}")
            
            col1, col2 = st.columns(2)
            with col1:
                # 수정: markdown에서 이모지 제거
                st.markdown("#### Top 3 강점")
                for strength in final_report['top_3_strengths']:
                    st.success(f"- {strength}")
            with col2:
                # 수정: markdown에서 이모지 제거
                st.markdown("#### Top 3 개선 과제")
                for priority in final_report['top_3_priorities']:
                    st.warning(f"- {priority}")
            
            # 수정: markdown에서 이모지 제거
            st.markdown("#### 주요 결정 필요 사안 (Decision Points)")
            for dp in final_report['decision_points']:
                st.error(f"- {dp}")
            
            st.markdown("---")
            # 수정: header에서 이모지 제거
            st.header("각 분야 별 상세 분석 결과")

            with st.expander("1. 정성 리뷰 분석 보고서 (텍스트 마이닝)"):
                report = expert_reports['qualitative']
                if report:
                    for cluster in report.get('keyword_clusters', []):
                        st.markdown(f"**- 테마:** {cluster.get('theme', 'N/A')} (**감성:** {cluster.get('sentiment', 'N/A')})")
                        st.caption(f"**주요 키워드:** {', '.join(cluster.get('keywords', []))}")
                        with st.container(border=True):
                            for example in cluster.get('review_examples', []):
                                st.write(f"> {example}")
                    # 수정: markdown에서 이모지 제거
                    st.markdown(f"**새로운 이슈:** {', '.join(report.get('emerging_issues', []))}")

            with st.expander("2. 강점 및 개선 우선순위 보고서 (전략 분석)"):
                report = expert_reports['strategic']
                if report:
                    # 수정: markdown에서 이모지 제거
                    st.markdown("#### 핵심 강점")
                    for strength in report.get('core_strengths', []):
                        st.markdown(f"- {strength}")
                    # 수정: markdown에서 이모지 제거
                    st.markdown("#### 개선 우선순위")
                    for priority in report.get('strategic_priorities', []):
                        st.markdown(f"**- 이슈:** {priority.get('issue', 'N/A')} (**예상 ROI:** {priority.get('expected_roi', 'N/A')})")
                        st.caption(f"**영향 분석:** {priority.get('impact_analysis', 'N/A')}")
                        st.caption(f"**개선 제안:** {priority.get('recommendation', 'N/A')}")
            
            with st.expander("3. 유저 세그먼트별 분석 보고서"):
                report = expert_reports['segmentation']
                if report:
                    for segment in report.get('user_segments', []):
                        # 수정: markdown에서 이모지 제거
                        st.markdown(f"#### {segment.get('segment_name', 'N/A')}")
                        st.write(f"**특징:** {segment.get('characteristics', 'N/A')}")
                        st.write(f"**주요 피드백:** {segment.get('feedback_summary', 'N/A')}")

            with st.expander("4. 향후 콘텐츠 및 전략 제안 보고서"):
                report = expert_reports['future']
                if report:
                    for proposal in report.get('proposals', []):
                        st.markdown(f"**- [{proposal.get('timeframe', 'N/A')}] {proposal.get('proposal_type', 'N/A')}:** {proposal.get('description', 'N/A')}")
                        st.caption(f"**제안 근거:** {proposal.get('rationale', 'N/A')}")
                    st.markdown("---")
                    # 수정: markdown에서 이모지 제거
                    st.markdown("#### 가설 기반 실험 제안 (A/B Test)")
                    st.info(f"{report.get('ab_test_suggestion', 'N/A')}")
        else:
            st.error("최종 보고서 생성에 실패했습니다.")


with tab2:
    # 수정: header에서 이모지 제거
    st.header("데이터 기반 현황 (Statistical Analysis)")
    st.info("""
    **분석 개요:** 이 탭은 제공된 리뷰 데이터를 기반으로 한 통계 분석 결과를 보여줍니다. 
    DAU, ARPU와 같은 운영 지표는 리뷰 데이터만으로 산출할 수 없으므로, 대신 리뷰어의 행동 패턴과 데이터 간의 관계를 심층적으로 분석합니다.
    """)
    st.subheader("1. 리뷰 기본 통계")
    col1, col2, col3, col4 = st.columns(4)
    positive_ratio = (df_filtered['긍정 리뷰'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    col1.metric("긍정 리뷰 비율", f"{positive_ratio:.1f}%")
    avg_playtime = df_filtered['playtime_forever_hours'].mean()
    col2.metric("평균 플레이 시간", f"{avg_playtime:.1f} 시간")
    avg_votes_up = df_filtered['votes_up'].mean()
    col3.metric("리뷰당 평균 '도움됨'", f"{avg_votes_up:.1f} 개")
    col4.metric("필터된 리뷰 수", f"{len(df_filtered)} 개")
    st.markdown("---")
    st.subheader("2. 플레이 시간 심층 분석")
    st.markdown("긍정 리뷰와 부정 리뷰를 남긴 유저 그룹 간의 **리뷰 작성 시점 플레이 시간** 분포를 비교합니다. 이를 통해 유저들이 주로 어느 플레이 단계에서 긍정/부정 경험을 하는지 추론할 수 있습니다.")
    if not df_filtered.empty:
        fig_box = px.box(df_filtered, x='긍정 리뷰', y='playtime_forever_hours', color='긍정 리뷰', points='all', title='긍정/부정 리뷰별 플레이 시간 분포', labels={'playtime_forever_hours': '플레이 시간 (시간)','긍정 리뷰': '리뷰 유형 (True: 긍정, False: 부정)'}, color_discrete_map={True: 'royalblue', False: 'darkorange'})
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("분석할 데이터가 없습니다.")
    st.markdown("---")
    st.subheader("3. 리뷰어 프로필 및 구매 경로 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**구매 경로별 리뷰 성향**")
        purchase_sentiment = df_filtered.groupby('steam_purchase')['긍정 리뷰'].value_counts(normalize=True).unstack().fillna(0)
        purchase_sentiment.index = purchase_sentiment.index.map({True: '스팀 상점 구매', False: '키 등록 등'})
        purchase_sentiment.columns = purchase_sentiment.columns.map({True: '긍정 리뷰', False: '부정 리뷰'})
        if not purchase_sentiment.empty:
            fig_purchase = px.bar(purchase_sentiment * 100, barmode='stack', title='구매 경로에 따른 긍정/부정 리뷰 비율', labels={'value': '비율 (%)', 'steam_purchase': '구매 경로', 'variable': '리뷰 유형'}, color_discrete_map={'긍정 리뷰': 'royalblue', '부정 리뷰': 'darkorange'}, text_auto='.2f')
            fig_purchase.update_traces(textangle=0, textposition="inside")
            st.plotly_chart(fig_purchase, use_container_width=True)
        else:
            st.warning("분석할 데이터가 없습니다.")
    with col2:
        st.markdown("**리뷰어의 보유 게임 수 분포**")
        if not df_filtered.empty:
            fig_games_owned = px.histogram(df_filtered, x='num_games_owned', nbins=50, title="리뷰어의 스팀 라이브러리 규모", labels={'num_games_owned': '보유 게임 수'})
            st.plotly_chart(fig_games_owned, use_container_width=True)
        else:
            st.warning("분석할 데이터가 없습니다.")
    st.markdown("---")
    st.subheader("4. 주요 지표 간 상관관계 분석 (Correlation Analysis)")
    if len(df_filtered) > 1:
        numerical_cols = ['playtime_forever_hours', 'votes_up', 'num_games_owned', 'num_reviews_by_author']
        corr_matrix = df_filtered[numerical_cols].corr()
        corr_matrix.rename(index={'playtime_forever_hours': '플레이 시간', 'votes_up': '도움됨 투표 수', 'num_games_owned': '보유 게임 수', 'num_reviews_by_author': '작성 리뷰 수'}, columns={'playtime_forever_hours': '플레이 시간', 'votes_up': '도움됨 투표 수', 'num_games_owned': '보유 게임 수', 'num_reviews_by_author': '작성 리뷰 수'}, inplace=True)
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1], title='주요 지표 간 상관관계 히트맵')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("상관관계 분석을 위해서는 2개 이상의 데이터가 필요합니다.")

with tab3:
    st.header("소수 의견 심층 분석")
    st.markdown("다수의 목소리에 가려진 중요한 인사이트를 담은 소수 의견을 발굴합니다.")

    if st.button("소수 의견 분석 실행하기", key="minority_analysis_button"):
        if 'reviews_for_analysis' in st.session_state:
            reviews = st.session_state['reviews_for_analysis']
            with st.spinner("AI가 숨은 보석을 찾고 있습니다..."):
                minority_report = run_expert_analysis(reviews, minority_opinion_prompt, MinorityReport)
                st.session_state['minority_report'] = minority_report
        else:
            st.warning("'AI 전문가 종합 분석' 탭에서 분석을 먼저 실행하여 분석 대상을 설정해주세요.")

    if 'minority_report' in st.session_state:
        report = st.session_state['minority_report']
        if report and report.get('minority_opinions'):
            for opinion in report['minority_opinions']:
                with st.container(border=True):
                    st.subheader(f"핵심 요약: {opinion.get('opinion_summary', 'N/A')}")
                    st.info(f"**잠재적 인사이트:** {opinion.get('potential_insight', 'N/A')}")
                    st.caption("대표 리뷰:")
                    st.write(f"> {opinion.get('reviewer_quote', 'N/A')}")
        else:
            st.info("분석 결과, 주목할 만한 소수 의견이 발견되지 않았습니다.")

with tab4:
    # 수정: header에서 이모지 제거
    st.header("원본 데이터 탐색")
    st.markdown("필터링된 리뷰 원본 데이터를 직접 확인하고 정렬하거나 검색할 수 있습니다.")
    st.dataframe(df_filtered[['리뷰 내용', '긍정 리뷰', 'playtime_forever_hours', 'votes_up', 'timestamp_created', 'steam_purchase']], use_container_width=True, height=600)