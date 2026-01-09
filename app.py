import os
import json
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ✅ page_config는 st import 직후, 딱 1번
st.set_page_config(page_title="성향 프로필 + 연애 상담 챗봇", page_icon="💬", layout="wide")
from streamlit.errors import StreamlitSecretNotFoundError

def get_secret(key: str, default=None):
    """secrets.toml이 없어도 터지지 않게 안전하게 읽기"""
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default


def require_password():
    # ✅ 배포에서만 비번: Streamlit Cloud에서 secrets에 APP_PASSWORD가 있을 때만 잠금
    app_pw = get_secret("APP_PASSWORD", None)
    if not app_pw:
        return True  # 로컬(또는 비번 미설정 배포)은 그냥 통과

    if st.session_state.get("authed", False):
        return True

    st.title("🔒 접근 비밀번호")
    pw = st.text_input("비밀번호를 입력하세요", type="password")
    if st.button("입장"):
        if pw == app_pw:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.stop()

require_password()

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 0) 경로/설정 (⭐ 여기만 수정)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

ENV_PATH = PROJECT_ROOT / ".env"
PERSONA_JSON_PATH = PROJECT_ROOT / "data" / "persona_rules.json"
PERSIST_ROOT = PROJECT_ROOT / "chroma_store"
DATA_DIR = str(PERSONA_JSON_PATH.parent)

# 폰트는 있으면 사용, 없으면 fallback
FONT_PATH = str(PROJECT_ROOT / "assets" / "Freesentation-6SemiBold.ttf")

COL_USER_PROFILE = "user_profile"
COL_COUNSEL_DB = "counsel_db"
COL_RISK_PROTOCOL = "risk_protocol"

PERSIST_USER = str(PERSIST_ROOT / COL_USER_PROFILE)
PERSIST_COUNSEL = str(PERSIST_ROOT / COL_COUNSEL_DB)
PERSIST_RISK = str(PERSIST_ROOT / COL_RISK_PROTOCOL)

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5-mini"
CUT = 4.5           # (기존 4.0 → 4.5)
GRAY = 0.35         # 애매 구간 폭(±)

def hi_lo(score: float) -> str:
    # 1~7 평균 기준: 4.5 이상이면 높음, 4.5 미만이면 낮음
    return "높음" if score >= CUT else "낮음"

def expr_style(expr_score: float) -> str:
    # 애매하면 상담에서만 참고하고, 결과는 한쪽으로 확정(강제 분류)
    # 원하시면 '중간형' 표시로 바꿀 수도 있습니다.
    return "표현형" if expr_score >= CUT else "억제형"
# ✅ 배포/로컬 겸용: 환경변수 주입
# - 로컬: .env 사용
# - 배포(Streamlit Cloud): st.secrets 사용
openai_key = get_secret("OPENAI_API_KEY", None)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

load_dotenv(dotenv_path=str(ENV_PATH))


# =========================================================
# 2) 폰트
# =========================================================
def get_font_prop(font_path: str) -> fm.FontProperties:
    try:
        if os.path.isfile(font_path):
            return fm.FontProperties(fname=font_path)
    except Exception:
        pass
    return fm.FontProperties()  # fallback


FP = get_font_prop(FONT_PATH)


# =========================================================
# 3) 공통 CSS
# =========================================================
st.markdown(
    """
<style>
/* 설문 질문 글씨 */
div[data-testid="stRadio"] > label {
    font-size: 20px;
    font-weight: 500;
}
/* 결과 제목 */
h1, h2, h3 {
    font-size: 32px !important;
}
/* 결과 설명 본문 */
div[data-testid="stMarkdownContainer"] p {
    font-size: 20px;
    line-height: 1.6;
}
/* 라디오 전체(문항) 텍스트 크기 */
div[data-testid="stRadio"] label {
    font-size: 20px !important;
}
/* 라디오 옵션(1~7) 숫자 크기 */
div[data-testid="stRadio"] div[role="radiogroup"] label span {
    font-size: 30px !important;
}
/* 라디오 동그라미(선택 원) 크기 */
div[data-testid="stRadio"] div[role="radiogroup"] label {
    transform: scale(1.15);
    margin-right: 60px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 4) 설문: TYPE_DB
# =========================================================
TYPE_DB = {
    ("안정형", "표현형", "높음"): {
        "emoji": "🦦",
        "name": "따뜻한 수달",
        "headline": "안정형 · 표현형 · 효능감 높음",
        "desc": (
            "이 유형은 자신과 타인을 모두 긍정적으로 보며, 감정을 자연스럽게 표현하는 편이에요. "
            "자신의 능력에 대한 확신도 있어 관계와 목표 모두에서 안정적인 기반을 가지고 움직입니다.\n\n"
            "갈등이 발생해도 과도하게 흔들리기보다 맥락을 설명하며 차분히 조율하려는 태도가 강해요.\n\n"
            "어려움이 와도 회피하기보다 해결을 향해 접근하고, “할 수 있다”는 믿음으로 조용히 지속해 나갑니다.\n\n"
            "모든 상황을 혼자 정리하려 하기보다, 필요할 땐 도움을 받아들이면 더 여유로운 관계를 만들 수 있어요."
        ),
    },
    ("안정형", "표현형", "낮음"): {
        "emoji": "🐑",
        "name": "잔잔한 양",
        "headline": "안정형 · 표현형 · 효능감 낮음",
        "desc": (
            "감정을 자연스럽게 드러내며 관계를 소중히 여기지만, 새로운 상황 앞에서는 자신감이 흔들릴 수 있어요.\n\n"
            "관계에서는 따뜻하고 일관된 분위기를 유지하지만, 도전 앞에서는 “내가 해낼 수 있을까”가 먼저 떠오를 때가 있어요.\n\n"
            "작은 성공 경험을 촘촘히 쌓고 스스로를 격려하는 습관이 생기면 행동 폭이 크게 넓어질 수 있습니다."
        ),
    },
    ("안정형", "억제형", "높음"): {
        "emoji": "🐻",
        "name": "숲을 지키는 곰",
        "headline": "안정형 · 억제형 · 효능감 높음",
        "desc": (
            "감정 표현은 절제하지만 자기·타인에 대한 긍정이 탄탄하고, 문제를 침착하게 해결하는 능력이 돋보입니다.\n\n"
            "다만 감정 공유가 적어 오해가 생길 수 있어요. 감정을 조금만 더 나누면 관계의 깊이가 훨씬 풍부해집니다."
        ),
    },
    ("안정형", "억제형", "낮음"): {
        "emoji": "🦊",
        "name": "바람을 지켜보는 사막여우",
        "headline": "안정형 · 억제형 · 효능감 낮음",
        "desc": (
            "타인에 대한 신뢰는 있으나 자기 확신은 조심스러운 편이에요. 감정을 억제하며 혼자 해결하려는 경향이 있습니다.\n\n"
            "점진적인 성공 경험을 쌓으면 자신감이 크게 상승하는 유형입니다."
        ),
    },
    ("불안형", "표현형", "높음"): {
        "emoji": "🐹",
        "name": "민감한 귀염둥이 햄스터",
        "headline": "불안형 · 표현형 · 효능감 높음",
        "desc": (
            "감정 반응은 빠르지만 ‘불안을 행동으로 전환’하는 힘이 있어 추진력이 강합니다.\n\n"
            "민감성을 단점이 아닌 ‘연결의 능력’으로 쓰는 방향이 도움이 됩니다."
        ),
    },
    ("불안형", "표현형", "낮음"): {
        "emoji": "🐦",
        "name": "감정 많은 참새",
        "headline": "불안형 · 표현형 · 효능감 낮음",
        "desc": (
            "감정 폭이 넓고 변화가 빠르며 인정 욕구가 강하지만 자기 확신이 약해 쉽게 흔들릴 수 있어요.\n\n"
            "감정을 깊이 느끼고 진심으로 관계를 대한다는 강점이 있으니, 스스로를 격려하는 연습이 관계 안정에 도움이 됩니다."
        ),
    },
    ("불안형", "억제형", "높음"): {
        "emoji": "🦌",
        "name": "고요한 사슴",
        "headline": "불안형 · 억제형 · 효능감 높음",
        "desc": (
            "불안을 예민하게 느끼지만 드러내지 않고 스스로 해결하려 합니다. 효능감이 높아 문제를 다루지만, 표현 억제로 거리감이 생길 수 있어요.\n\n"
            "안전한 방식의 감정 표현 연습이 큰 도움이 됩니다."
        ),
    },
    ("불안형", "억제형", "낮음"): {
        "emoji": "🐱",
        "name": "숨어 있는 고양이",
        "headline": "불안형 · 억제형 · 효능감 낮음",
        "desc": (
            "불안은 크지만 표현은 조용해 내부 스트레스가 오래 쌓일 수 있어요.\n\n"
            "감정을 안전하게 나누는 경험 + 작은 성취 반복이 매우 중요합니다."
        ),
    },
    ("회피형", "표현형", "높음"): {
        "emoji": "🐺",
        "name": "떠돌이 늑대",
        "headline": "회피형 · 표현형 · 효능감 높음",
        "desc": (
            "표현은 자연스럽지만 깊은 관계엔 조심스러울 수 있어요. 혼자 해결 능력은 강합니다.\n\n"
            "관계를 끊기보다 ‘경계를 조절하는 기술’을 익히면 균형이 좋아집니다."
        ),
    },
    ("회피형", "표현형", "낮음"): {
        "emoji": "🐨",
        "name": "하품하는 코알라",
        "headline": "회피형 · 표현형 · 효능감 낮음",
        "desc": (
            "관계가 깊어질 때 불안과 회피가 동시에 올라올 수 있어요.\n\n"
            "안정적인 성공경험과 지지 경험을 천천히 쌓는 것이 변화의 핵심입니다."
        ),
    },
    ("회피형", "억제형", "높음"): {
        "emoji": "🐆",
        "name": "독립적인 표범",
        "headline": "회피형 · 억제형 · 효능감 높음",
        "desc": (
            "표현은 절제하고 관계는 조심스럽게 유지하지만, 혼자 해결에 강합니다.\n\n"
            "감정을 드러내는 것이 약점이라는 신념을 내려놓는 순간 관계의 질이 달라집니다."
        ),
    },
    ("회피형", "억제형", "낮음"): {
        "emoji": "🐢",
        "name": "바위 틈의 거북",
        "headline": "회피형 · 억제형 · 효능감 낮음",
        "desc": (
            "자기 확신이 낮고 타인을 쉽게 신뢰하지 않아 관계와 도전 모두에 조심스럽습니다.\n\n"
            "작게 구조화된 목표부터 성공경험을 쌓는 접근이 잘 맞습니다."
        ),
    },
    ("거부형", "표현형", "높음"): {
        "emoji": "🦅",
        "name": "대담한 매",
        "headline": "거부형 · 표현형 · 효능감 높음",
        "desc": (
            "자기 믿음은 강하지만 타인 신뢰는 낮아 관계는 가볍게 유지될 수 있어요.\n\n"
            "깊은 정서 교류가 부담스러울 때, ‘가능한 범위’부터 합의하는 방식이 유효합니다."
        ),
    },
    ("거부형", "표현형", "낮음"): {
        "emoji": "🐦‍⬛",
        "name": "새벽 까마귀",
        "headline": "거부형 · 표현형 · 효능감 낮음",
        "desc": (
            "표현은 하지만 타인 신뢰가 낮아 일정 선을 유지하려는 경향이 있습니다.\n\n"
            "작은 성공경험을 쌓아 ‘할 수 있다’ 감각을 회복하는 게 중요합니다."
        ),
    },
    ("거부형", "억제형", "높음"): {
        "emoji": "🐈‍⬛",
        "name": "고독한 전략가 흑호",
        "headline": "거부형 · 억제형 · 효능감 높음",
        "desc": (
            "감정 표현은 거의 없고 자기 지탱 힘이 강합니다.\n\n"
            "표현을 조금만 허용하면 연결이 부드러워지고 피로감이 줄 수 있어요."
        ),
    },
    ("거부형", "억제형", "낮음"): {
        "emoji": "🦉",
        "name": "고목 위 부엉이",
        "headline": "거부형 · 억제형 · 효능감 낮음",
        "desc": (
            "자기·타인 긍정 모두 낮아 관계를 매우 신중하게 대합니다.\n\n"
            "작은 성공 경험 + 신뢰할 수 있는 한 사람의 확보가 회복에 큰 도움이 됩니다."
        ),
    },
}


def get_type_info(base: str, style: str, eff: str) -> Dict[str, Any]:
    return TYPE_DB.get(
        (base, style, eff),
        {"emoji": "🐾", "name": "임시 유형", "headline": f"{base} · {style} · 효능감 {eff}", "desc": "이 유형 설명은 준비 중입니다."},
    )


# =========================================================
# 5) 설문 문항 구성
# =========================================================
def rev7(x: int) -> int:
    return 8 - x


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 4.0


def safe_mean(xs: List[Optional[float]]) -> float:
    xs2 = [x for x in xs if x is not None]
    return sum(xs2) / len(xs2) if xs2 else 0.0


def internal_ratio(pos_vals: List[float], neg_raw_vals: List[float], eps: float = 1e-9) -> float:
    """pos / (pos + neg)  -> 0~1"""
    P = safe_mean(pos_vals)
    N = safe_mean(neg_raw_vals)
    return P / (P + N + eps)


def base_type(self_m_pct: float, other_m_pct: float) -> str:
    x = self_m_pct >= 50
    y = other_m_pct >= 50
    if x and y:
        return "안정형"
    if (not x) and y:
        return "불안형"
    if x and (not y):
        return "회피형"
    return "거부형"


QUESTIONS: List[Dict[str, Any]] = []

# -----------------------------
# 1) Self Model (5문항)
# - pos 3개, neg 2개(역채점)
# -----------------------------
self_pos_texts = [
    "실수해도 ‘내가 무가치해진 건 아니다’라고 비교적 빨리 정리하는 편이다.",
    "중요한 결정을 앞두면, 결국은 내가 감당할 수 있다는 쪽에 더 무게가 실린다.",
    "비판을 들어도, 내 전체를 부정당한 느낌보단 ‘부분 피드백’으로 받아들이려 한다.",
]
self_neg_texts = [
    "상대 반응이 차가우면 ‘내가 문제라서’라는 해석이 먼저 떠오르는 편이다.",
    "사랑받으려면 ‘지금의 나’로는 부족하다는 생각이 종종 든다.",
]

for i, t in enumerate(self_pos_texts, start=1):
    QUESTIONS.append({"key": f"s{i}", "text": t, "scale": "self_pos", "reverse": False})
for j, t in enumerate(self_neg_texts, start=4):
    QUESTIONS.append({"key": f"s{j}", "text": t, "scale": "self_neg", "reverse": True})


# -----------------------------
# 2) Other Model (5문항)
# - pos 3개, neg 2개(역채점)
# -----------------------------
other_pos_texts = [
    "도움을 요청하면, 대체로 사람들은 나를 해치기보다 도우려 했던 경험이 더 많다.",
    "관계가 깊어질수록 ‘연결이 생긴다’는 기대가 비교적 자연스럽다.",
    "내가 솔직히 말해도, 상대가 전부 공격으로 받진 않을 거라고 생각하는 편이다.",
]
other_neg_texts = [
    "가까워질수록 ‘언젠가 상처받을 것 같다’는 경계가 먼저 올라오는 편이다.",
    "호의를 받아도 ‘속에 다른 의도가 있을 수 있다’는 의심이 스치는 편이다.",
]

for i, t in enumerate(other_pos_texts, start=1):
    QUESTIONS.append({"key": f"o{i}", "text": t, "scale": "other_pos", "reverse": False})
for j, t in enumerate(other_neg_texts, start=4):
    QUESTIONS.append({"key": f"o{j}", "text": t, "scale": "other_neg", "reverse": True})


# -----------------------------
# 3) Emotion Reg (6문항)
# - 표현(역채점한 억제) 3문항
# - 재평가 3문항(참고용, 분류에는 직접 안 쓰더라도 상담에 도움됨)
# -----------------------------
expr_supp_texts = [
    "감정이 커져도 ‘티 안 나게’ 정리하려는 편이다.",
    "좋아도 싫어도 표정/말투가 크게 드러나지 않게 조절하는 편이다.",
    "갈등이 생기면 감정을 말하기보다 일단 눌러두고 넘어가려 한다.",
]
reapp_texts = [
    "기분이 가라앉으면, 일부러 의미/좋은 점을 찾아 해석을 바꿔보는 편이다.",
    "상대 말에 상처받아도 ‘그럴 수도 있지’로 마음을 정리하려 한다.",
    "스트레스가 오면, 상황을 더 차분한 관점으로 다시 보는 편이다.",
]

# 억제문항을 역채점해서 '표현 점수'로 만듦
for i, t in enumerate(expr_supp_texts, start=1):
    QUESTIONS.append({"key": f"e{i}", "text": t, "scale": "erq_expr", "reverse": True})
for i, t in enumerate(reapp_texts, start=4):
    QUESTIONS.append({"key": f"e{i}", "text": t, "scale": "erq_reapp", "reverse": False})


# -----------------------------
# 4) Self-efficacy (6문항)
# -----------------------------
eff_texts = [
    "불안해도 ‘일단 해보자’로 시작하는 편이다.",
    "막히면 포기보다 ‘다른 방법’을 찾아보는 쪽이 더 빠르다.",
    "실패해도 ‘내 능력 전체’로 일반화하기보다 다음 시도를 준비하는 편이다.",
    "조언을 들으면 ‘비판’보다 ‘업그레이드 기회’로 받아들이려 한다.",
    "부담이 커도 도망치기보다 ‘작게 쪼개서’ 처리하려 한다.",
    "긴장해도 해야 할 일의 핵심만 잡고 계속 진행할 수 있는 편이다.",
]
for i, t in enumerate(eff_texts, start=1):
    QUESTIONS.append({"key": f"g{i}", "text": t, "scale": "eff", "reverse": False})


def get_vals(scale: str, answers: Dict[str, int]) -> List[int]:
    vals: List[int] = []
    for q in QUESTIONS:
        if q["scale"] == scale:
            v = answers.get(q["key"], 4)
            if q["reverse"]:
                v = rev7(v)
            vals.append(v)
    return vals


def get_vals_raw(scale: str, answers: Dict[str, int]) -> List[int]:
    vals: List[int] = []
    for q in QUESTIONS:
        if q["scale"] == scale:
            vals.append(answers.get(q["key"], 4))
    return vals


# =========================================================
# 6) persona_rules + RAG 유틸
# =========================================================
SYSTEM_POLICY = """
[챗봇 정체성]
본 챗봇은 연애 및 관계에 대한 고민을 함께 정리하는 AI 상담 파트너이며,
전문 상담사·의료·법률 전문가가 아닙니다.
모든 조언은 참고용 관점 제시에 해당합니다.

[상담 원칙 / 윤리 기준]
1. 관계 갈등을 옳고 그름의 문제로 판단하지 않고, 욕구·기대·상황의 충돌로 해석합니다.
2. 감정은 평가하지 않고 이해의 대상으로 다루며, 감정보다 감정을 다루는 방식에 주목합니다.
3. 과도한 희생이나 집착을 관계의 건강 신호로 해석하지 않습니다.
4. 자율성을 관계의 위협이 아닌 핵심 요소로 존중합니다.
5. 제한된 정보로 상대의 의도·성격·관계를 단정하지 않습니다.
6. 공감하되, 감정에서 비롯된 모든 행동을 정당화하지 않습니다.
7. 빠른 결론보다 사고의 확장과 맥락 이해를 우선합니다.
8. 의료적·법적 조언이나 진단을 하지 않으며, 사용자의 선택을 대신 결정하거나 강요하지 않습니다.
9. 윤리적·관계적 위험이 있는 요청은 수행하지 않으며, 대화를 더 안전한 방향으로 전환합니다.
10. 공감은 사실 기반으로 유지하고, 한쪽에 치우치지 않는 중립적 균형을 지킵니다.
11. 실명·연락처·위치 등 민감한 정보를 요구하거나 활용하지 않습니다.

[안전 대응 원칙]
- 자해·자살·폭력·즉각적 안전 위협 신호가 감지될 경우, 공감과 안전 확보를 최우선으로 안내합니다.
- 불법·감시·통제·조작을 돕는 구체적 방법은 제공하지 않습니다.
""".strip()

RISK_BADGE = "🚨 위험신호 발견"
RISK_PATTERNS = [
    r"자해", r"자살", r"죽고\s*싶", r"살\s*의미", r"폭력", r"때리", r"죽여",
    r"스토킹", r"위치\s*추적", r"감시", r"통제", r"협박", r"가스라이팅",
    r"숨이\s*막혀", r"패닉", r"공황", r"아무것도\s*못\s*하겠",
]

SUMMARY_LABELS = ["[감정]", "[핵심 고민]", "[오늘 정리된 방향]", "[다음 한 걸음]", "[안전/경계]"]

FINAL_SUMMARY_FORMAT = """\
[감정] ...
[핵심 고민] ...
[오늘 정리된 방향] ...
[다음 한 걸음] ...\
"""

FINAL_SUMMARY_FORMAT_WITH_SAFETY = """\
[감정] ...
[핵심 고민] ...
[오늘 정리된 방향] ...
[다음 한 걸음] ...
[안전/경계] ...\
"""

FEW_SHOT_EXAMPLES = [
    {
        "history_summary": "연인이 바쁠 때 연락이 줄어 불안해짐. 추궁하면 갈등이 커질까 걱정함. 상대는 여유가 부족한 상황일 가능성이 큼.",
        "risk_mode": False,
        "output": "\n".join([
            "[감정] 서운함과 불안이 함께 올라오셨습니다.",
            "[핵심 고민] 연락 빈도를 애정으로 해석하게 되면서 마음이 흔들리는 점이 핵심입니다.",
            "[오늘 정리된 방향] 추궁 대신 ‘필요한 연결 방식’을 구체적으로 합의하는 쪽이 안전합니다.",
            "[다음 한 걸음] 오늘은 추가 메시지를 멈추고, 내일 10분 통화 루틴을 제안해 보세요.",
        ])
    },
    {
        "history_summary": "상대가 위치 추적을 원하거나 감시/통제를 요구하는 맥락이 있었고, 사용자가 불안을 크게 느낌. 안전과 경계가 우선 필요함.",
        "risk_mode": True,
        "output": "\n".join([
            "[감정] 불안과 압박감이 크게 느껴지셨습니다.",
            "[핵심 고민] 관계에서 ‘통제/감시’가 안전감을 해치고 있습니다.",
            "[오늘 정리된 방향] 상대의 요구를 즉시 수용하기보다 경계를 명확히 세우는 것이 우선입니다.",
            "[다음 한 걸음] 위치/비밀번호 공유는 중단하고, ‘이건 불편해서 못 한다’는 한 문장만 전달하세요.",
            "[안전/경계] 위협·협박이 느껴지면 주변 도움(지인/기관)으로 안전을 먼저 확보하세요.",
        ])
    },
]


def detect_risk_mode(user_message: str) -> bool:
    return any(re.search(p, user_message or "") for p in RISK_PATTERNS)


@st.cache_resource(show_spinner=False)
def load_persona_rules_cached(data_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_dir, "persona_rules.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("persona_rules.json은 비어있지 않은 list여야 합니다.")
    return data


def pick_persona_rule_from_json(profile: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    nickname = (profile.get("nickname") or "").strip()
    attachment = (profile.get("attachment_type") or profile.get("attachment") or "").strip()
    emotion_reg = (profile.get("emotion_reg") or "").strip()
    efficacy = (profile.get("efficacy") or profile.get("self_efficacy") or "").strip()

    if nickname:
        for r in rules:
            if (r.get("nickname") or "").strip() == nickname:
                return r

    if attachment or emotion_reg or efficacy:
        for r in rules:
            axis = r.get("axis") or {}
            ok = True
            if attachment and axis.get("attachment") != attachment:
                ok = False
            if emotion_reg and axis.get("emotion_reg") != emotion_reg:
                ok = False
            if efficacy and axis.get("efficacy") != efficacy:
                ok = False
            if ok:
                return r

    return rules[0]


def make_counselor_state_from_rule(rule: Dict[str, Any]) -> str:
    forbidden = rule.get("forbidden_phrases") or []
    forbidden_str = ", ".join(forbidden) if isinstance(forbidden, list) else str(forbidden)
    return f"""
[상담자 운영 상태 / counselor_state]
- 페르소나(별명): {rule.get("nickname", "")}
- 권장 톤: {rule.get("tone", "동등·존중형")}
- 상담자 목표: {rule.get("goal", "사용자 부담 완화 + 현실적 조율")}
- 핵심 특성(주의점): {rule.get("core_traits", "감정 안정/균형 유지")}
- 금지 화법(절대 사용 금지): {forbidden_str if forbidden_str else "상대/사용자 비난, 강요, 단정"}
""".strip()


@st.cache_resource(show_spinner=True)
def load_vectorstores_only() -> Dict[str, Chroma]:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    for p in (PERSIST_USER, PERSIST_COUNSEL, PERSIST_RISK):
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"persist_directory not found: {p}\n"
                f"먼저 ingest(적재) 스크립트로 VectorDB를 생성/저장하세요."
            )

    user_profile_db = Chroma(
        collection_name=COL_USER_PROFILE,
        persist_directory=PERSIST_USER,
        embedding_function=embeddings,
    )
    counsel_db = Chroma(
        collection_name=COL_COUNSEL_DB,
        persist_directory=PERSIST_COUNSEL,
        embedding_function=embeddings,
    )
    risk_db = Chroma(
        collection_name=COL_RISK_PROTOCOL,
        persist_directory=PERSIST_RISK,
        embedding_function=embeddings,
    )
    return {"user_profile_db": user_profile_db, "counsel_db": counsel_db, "risk_db": risk_db}


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.6)


def build_query(history_summary: str, user_message: str) -> str:
    return (history_summary.strip() + "\n" + user_message.strip()).strip()


def get_counsel_context(counsel_db: Chroma, history_summary: str, user_message: str, k: int = 4) -> str:
    q = build_query(history_summary, user_message)
    docs = counsel_db.similarity_search(q, k=k, filter={"doc_type": "playbook"})
    return "\n\n---\n\n".join([d.page_content for d in docs]).strip()


def parse_required_steps_from_text(page_content: str) -> List[str]:
    m = re.search(r"\[필수Step\]\s*(.+)", page_content)
    if not m:
        return []
    raw = m.group(1).strip()
    parts = [p.strip() for p in re.split(r"[→>\-]|,", raw) if p.strip()]
    out: List[str] = []
    for p in parts:
        mm = re.search(r"step\s*([0-9]+)", p, flags=re.IGNORECASE)
        if mm:
            out.append(f"STEP_{mm.group(1)}")
    return out


def select_risk_level_doc(risk_db: Chroma, history_summary: str, user_message: str, k: int = 3):
    q = build_query(history_summary, user_message)
    docs = risk_db.similarity_search(q, k=k, filter={"doc_type": "risk_level_example"})
    if not docs:
        docs = risk_db.similarity_search(q, k=k, filter={"doc_type": "risk_response_map"})
    return docs[0]


def get_required_steps(level_doc) -> List[str]:
    md = level_doc.metadata or {}
    rs = md.get("required_steps")
    if isinstance(rs, list) and rs:
        if len(rs) == 1 and isinstance(rs[0], str) and "Step" in rs[0]:
            return parse_required_steps_from_text(f"[필수Step] {rs[0]}")
        out = [x.upper() for x in rs if isinstance(x, str) and x.upper().startswith("STEP_")]
        if out:
            return out
    return parse_required_steps_from_text(level_doc.page_content)


def fetch_risk_steps_context(risk_db: Chroma, step_ids: List[str]) -> str:
    blocks: List[str] = []
    for sid in step_ids:
        try:
            docs = risk_db.similarity_search(
                query=f"{sid} risk step",
                k=2,
                filter={"doc_type": "risk_step", "step_id": sid},
            )
        except Exception:
            docs = []
        if not docs:
            docs = risk_db.similarity_search(query=f"{sid} 단계", k=2, filter={"doc_type": "risk_step"})
            docs = docs[:1]
        blocks.extend([d.page_content for d in docs[:1]])
    return "\n\n---\n\n".join(blocks).strip()


def extract_level(md: Dict[str, Any]) -> str:
    keys = md.get("keys")
    if isinstance(keys, dict):
        lvl = keys.get("level")
        if lvl is not None:
            return str(lvl)
    if isinstance(keys, str):
        try:
            parsed = json.loads(keys)
            if isinstance(parsed, dict) and parsed.get("level") is not None:
                return str(parsed.get("level"))
        except Exception:
            pass
    if md.get("level") is not None:
        return str(md.get("level"))
    if md.get("row_id") is not None:
        return str(md.get("row_id"))
    return "UNKNOWN"


def build_risk_pack(risk_db: Chroma, history_summary: str, user_message: str) -> Dict[str, Any]:
    level_doc = select_risk_level_doc(risk_db, history_summary, user_message)
    required_steps = get_required_steps(level_doc)
    t07 = fetch_risk_steps_context(risk_db, required_steps)

    md = level_doc.metadata or {}
    level = extract_level(md)

    return {
        "level": level,
        "required_steps": required_steps,
        "t06_context": level_doc.page_content,
        "t07_context": t07,
    }


def generate_answer(
    llm: ChatOpenAI,
    counselor_state: str,
    counsel_context: str,
    risk_mode: bool,
    risk_pack: Optional[Dict[str, Any]],
    history_summary: str,
    user_message: str,
) -> str:
    risk_block = ""
    if risk_mode and risk_pack:
        risk_block = f"""
[위험 대응 가이드 / risk_pack]
- 선택된 Level: {risk_pack.get("level")}
- 필수 Step: {", ".join(risk_pack.get("required_steps", []))}
- t06(Level 문서):
{risk_pack.get("t06_context","")}

- t07(Step 문서):
{risk_pack.get("t07_context","")}
""".strip()

    prompt = f"""
{SYSTEM_POLICY}

{counselor_state}

[참고 컨텍스트 / counsel_context]
{counsel_context}

{risk_block}

[대화 요약 / history_summary]
{history_summary}

[최신 사용자 발화 / user_message]
{user_message}

[지시]
- 금지 화법은 절대 사용하지 마세요.
- risk_mode={risk_mode}인 경우, Step 흐름을 답변 구조에 반영하세요.
- 다음 한 걸음(질문 1~2개 또는 행동 1~2개)을 포함하세요.
- 답변은 3~4줄 이내로 작성하세요. 목록형 설명 금지.
- 항상 존댓말 사용하세요.
""".strip()

    answer = (llm.invoke(prompt).content or "").strip()
    if risk_mode:
        answer = f"{RISK_BADGE}\n\n{answer}"
    return answer


def update_history_summary(llm: ChatOpenAI, prev_summary: str, user_message: str, assistant_answer: str) -> str:
    prompt = f"""
아래 정보를 바탕으로 '대화 요약'을 3~5줄 한국어로 갱신하세요.

[이전 요약]
{prev_summary}

[사용자 발화]
{user_message}

[상담자 답변]
{assistant_answer}

[출력]
- 3~5줄 요약(줄바꿈 포함)
""".strip()
    return (llm.invoke(prompt).content or "").strip()


def enforce_linebreaks(text: str) -> str:
    t = (text or "").strip()
    for lab in SUMMARY_LABELS:
        t = t.replace(lab, f"\n{lab}")
    t = t.lstrip("\n")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return "\n".join(lines)


def final_summary_fewshot(llm: ChatOpenAI, history_summary: str, risk_mode: bool) -> str:
    shots_txt = []
    for ex in FEW_SHOT_EXAMPLES:
        shots_txt.append(
            "### 예시 입력\n"
            f"[대화 요약]\n{ex['history_summary']}\n"
            f"[risk_mode]\n{ex['risk_mode']}\n\n"
            "### 예시 출력(정답 형식)\n"
            f"{ex['output']}\n"
        )
    shots_block = "\n\n".join(shots_txt).strip()
    format_block = FINAL_SUMMARY_FORMAT_WITH_SAFETY if risk_mode else FINAL_SUMMARY_FORMAT

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 연애/관계 상담 대화를 ‘상담 종료 요약’으로 정리하는 도우미입니다.\n"
         "반드시 사용자가 읽기 쉬운 한국어 존댓말로만 작성하세요.\n"
         "절대 목록(불릿/번호)을 쓰지 말고, 아래 지정 양식 그대로 줄바꿈을 유지하세요.\n"
         "출력은 오직 요약 본문만 반환하세요(설명/서문/코드 금지)."),
        ("human",
         f"[지정 양식]\n{format_block}\n\n"
         f"[few-shot 예시]\n{shots_block}\n\n"
         f"[실제 입력]\n[대화 요약]\n{history_summary}\n[risk_mode]\n{risk_mode}\n\n"
         "[작성 규칙]\n"
         "- 총 3~5줄(위험모드면 4~5줄)\n"
         "- 각 줄은 양식의 라벨로 시작\n"
         "- 조언은 ‘다음 한 걸음’에만 1줄로\n"
         "- risk_mode=True면 [안전/경계] 줄을 반드시 포함\n"
         )
    ])

    text = (llm.invoke(prompt.format_messages()).content or "").strip()
    return enforce_linebreaks(text)


def run_turn(
    llm: ChatOpenAI,
    persona_rule: Dict[str, Any],
    counsel_db: Chroma,
    risk_db: Chroma,
    history_summary: str,
    user_message: str,
) -> Dict[str, Any]:
    counselor_state = make_counselor_state_from_rule(persona_rule)
    counsel_context = get_counsel_context(counsel_db, history_summary, user_message, k=4)

    risk_mode = detect_risk_mode(user_message)
    risk_pack = build_risk_pack(risk_db, history_summary, user_message) if risk_mode else None

    assistant_answer = generate_answer(
        llm, counselor_state, counsel_context, risk_mode, risk_pack, history_summary, user_message
    )
    new_summary = update_history_summary(llm, history_summary, user_message, assistant_answer)

    return {"assistant_answer": assistant_answer, "history_summary": new_summary, "risk_mode": risk_mode}


# =========================================================
# 7) 설문 UI
# =========================================================
def init_survey_state():
    if "survey_page" not in st.session_state:
        st.session_state.survey_page = "survey"  # survey/result
    if "order" not in st.session_state:
        order = list(range(len(QUESTIONS)))
        random.shuffle(order)
        st.session_state.order = order
    if "survey_completed" not in st.session_state:
        st.session_state.survey_completed = False
    if "survey_answers" not in st.session_state:
        st.session_state.survey_answers = None


def init_chat_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history_summary" not in st.session_state:
        st.session_state.history_summary = "상담 시작. 초기 맥락 파악 단계."
    if "persona_rule" not in st.session_state:
        st.session_state.persona_rule = None
    if "ever_risk" not in st.session_state:
        st.session_state.ever_risk = False


def go_survey():
    st.session_state.mode = "survey"
    st.session_state.survey_page = "survey"
    st.rerun()


def go_result():
    st.session_state.survey_page = "result"
    st.rerun()


def go_chat():
    st.session_state.mode = "chat"
    st.rerun()


def reset_survey_answers():
    for q in QUESTIONS:
        if q["key"] in st.session_state:
            del st.session_state[q["key"]]
    if "order" in st.session_state:
        del st.session_state["order"]
    st.session_state.survey_page = "survey"
    st.session_state.survey_completed = False
    st.session_state.survey_answers = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.history_summary = "상담 시작. 초기 맥락 파악 단계."
    st.session_state.ever_risk = False


def render_survey():
    st.title("성향 프로필 (설문)")
    st.caption("1(전혀 아니다) ~ 7(매우 그렇다)")

    for i, idx in enumerate(st.session_state.order, start=1):
        q = QUESTIONS[idx]
        st.markdown(f"**{i}. {q['text']}**")
        CHOICES_NO_MID = [1, 2, 3, 5, 6, 7]

        st.radio(
            label="",
            options=CHOICES_NO_MID,
            horizontal=True,
            key=q["key"],
        )
        st.markdown("---")

    if st.button("다음 ▶ (결과 보기)", use_container_width=True):
    # ✅ 설문 응답 스냅샷 저장(다시보기용)
        st.session_state.survey_answers = {q["key"]: st.session_state.get(q["key"], 4) for q in QUESTIONS}
        st.session_state.survey_completed = True
        go_result()



def draw_quadrant(self_model: float, other_model: float):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axvline(50, color="black", linewidth=2.0, zorder=1)
    ax.axhline(50, color="black", linewidth=2.0, zorder=1)

    ax.text(25, 75, "안정형", ha="center", va="center", fontproperties=FP, fontsize=16)
    ax.text(75, 75, "의존형", ha="center", va="center", fontproperties=FP, fontsize=16)
    ax.text(25, 25, "거부형", ha="center", va="center", fontproperties=FP, fontsize=16)
    ax.text(75, 25, "회피형", ha="center", va="center", fontproperties=FP, fontsize=16)

    ax.text(-10, 50, "타인에\n대한\n생각", ha="center", va="center", fontproperties=FP, fontsize=18, rotation=90)
    ax.text(-10, 92, "긍정적", ha="center", va="center", fontproperties=FP, fontsize=12, rotation=90)
    ax.text(-10, 8, "부정적", ha="center", va="center", fontproperties=FP, fontsize=12, rotation=90)

    ax.text(50, -12, "자신에\n대한\n생각", ha="center", va="center", fontproperties=FP, fontsize=18)
    ax.text(8, -12, "부정적", ha="left", va="center", fontproperties=FP, fontsize=12)
    ax.text(92, -12, "긍정적", ha="right", va="center", fontproperties=FP, fontsize=12)

    ax.scatter([self_model], [other_model], s=260, color="#F28C28", edgecolors="white", linewidths=2.5, zorder=3)

    plt.tight_layout()
    st.pyplot(fig)


def score_to_pct_0_100(score_1_7: float) -> int:
    return int(round((score_1_7 - 1) / 6 * 100))


def draw_dual_bar(ax, pct_right, left_end_label, right_end_label, title, font_prop):
    """
    pct_right: 0~100 (오른쪽 비율)
    - 오른쪽이 '높음/표현' 같이 긍정 방향일 때 그대로 넣으면 직관적으로 맞습니다.
    """
    pct_right = max(0, min(100, int(pct_right)))
    pct_left = 100 - pct_right

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bar_h = 0.42
    y = 0.5

    # 왼쪽(낮음/억제), 오른쪽(높음/표현)
    left_color = "#D7EAF6"
    right_color = "#F28C28"

    # 왼쪽 채움
    ax.barh([y], [pct_left], height=bar_h, left=0, zorder=1, color=left_color)
    # 오른쪽 채움
    ax.barh([y], [pct_right], height=bar_h, left=pct_left, zorder=2, color=right_color)

    ax.text(0, 1.05, title, ha="left", va="bottom", fontproperties=font_prop, fontsize=14)

    ax.text(0, -0.15, left_end_label, ha="left", va="top", fontproperties=font_prop, fontsize=11)
    ax.text(100, -0.15, right_end_label, ha="right", va="top", fontproperties=font_prop, fontsize=11)

    # 왼쪽 % 표시
    if pct_left >= 10:
        ax.text(pct_left - 2, y, f"{pct_left}%", ha="right", va="center",
                fontproperties=font_prop, fontsize=13, color="#2B2B2B", zorder=3)
    else:
        ax.text(pct_left + 2, y, f"{pct_left}%", ha="left", va="center",
                fontproperties=font_prop, fontsize=13, color="#2B2B2B", zorder=3)

    # 오른쪽 % 표시
    if pct_right >= 10:
        ax.text(98, y, f"{pct_right}%", ha="right", va="center",
                fontproperties=font_prop, fontsize=13, color="white", zorder=3)
    else:
        ax.text(pct_left + 2, y, f"{pct_right}%", ha="left", va="center",
                fontproperties=font_prop, fontsize=13, color="white", zorder=3)


def render_result():
    st.title("성향 프로필 (결과)")

    if st.session_state.get("survey_completed") and st.session_state.get("survey_answers"):
        answers = st.session_state.survey_answers
    else:
        answers = {q["key"]: st.session_state.get(q["key"], 4) for q in QUESTIONS}

    self_pos = get_vals("self_pos", answers)
    self_neg_raw = get_vals_raw("self_neg", answers)
    other_pos = get_vals("other_pos", answers)
    other_neg_raw = get_vals_raw("other_neg", answers)

    self_model = internal_ratio(self_pos, self_neg_raw) * 100
    other_model = internal_ratio(other_pos, other_neg_raw) * 100

    expression = mean(get_vals("erq_expr", answers))
    efficacy = mean(get_vals("eff", answers))

    base = base_type(self_model, other_model)
    style = expr_style(expression)

    eff = hi_lo(efficacy)


    info = get_type_info(base, style, eff)

    st.subheader(f"{info['emoji']} {info['name']}")
    st.caption(info["headline"])
    st.write(info["desc"])

    with st.expander("그래프 보기", expanded=False):
        st.write("애착 영역")
        draw_quadrant(self_model, other_model)

        # 높을수록 오른쪽(표현/높음)로 가는 점수
        expr_pct = score_to_pct_0_100(expression)  # 높을수록 '표현'
        eff_pct = score_to_pct_0_100(efficacy)     # 높을수록 '자기효능감 높음'

        # ✅ 왼쪽이 '억제'이므로, 왼쪽(억제) 비율 = 100 - expr_pct
        fig1, ax1 = plt.subplots(figsize=(7.2, 1.1))
        draw_dual_bar(
            ax1,
            100 - expr_pct,
            "억제",
            "표현",
            "억제 ↔ 표현 (표현 점수)",
            FP,
        )
        st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        # ✅ 왼쪽이 '낮음'이므로, 왼쪽(낮음) 비율 = 100 - eff_pct
        fig2, ax2 = plt.subplots(figsize=(7.2, 1.1))
        draw_dual_bar(
            ax2,
            100 - eff_pct,
            "자기효능감 낮음",
            "자기효능감 높음",
            "자기효능감",
            FP,
        )
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

    st.divider()

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("◀ 설문 다시", use_container_width=True):
            st.session_state.survey_page = "survey"
            st.rerun()

    with colB:
        if st.button("응답 초기화", use_container_width=True):
            reset_survey_answers()
            st.rerun()

    with colC:
        if st.button("이 프로필로 상담 시작 💬", use_container_width=True):
            # ✅ 설문 결과 → 표준 profile 저장
            st.session_state.profile = {
                "attachment_type": base,  # "안정형/불안형/회피형/거부형"
                "emotion_reg": style,     # "표현형/억제형"
                "efficacy": eff,          # "높음/낮음"
                "nickname": "",           # axis 기반 자동매칭 사용
            }

            # ✅ persona_rules 자동 매칭
            persona_rules = load_persona_rules_cached(DATA_DIR)
            st.session_state.persona_rule = pick_persona_rule_from_json(st.session_state.profile, persona_rules)

            # ✅ 챗봇 상태 초기화
            init_chat_state()
            st.session_state.initialized = True
            reset_chat()

            go_chat()


# =========================================================
# 8) 챗봇 UI
# =========================================================
def render_chat():
    st.title("💬 연애/관계 상담 챗봇")

    profile = st.session_state.get("profile")
    persona_rule = st.session_state.get("persona_rule")

    if profile:
        st.caption(f"프로필: 애착={profile.get('attachment_type')} · 감정조절={profile.get('emotion_reg')} · 효능감={profile.get('efficacy')}")

    # 상단 버튼
    # 상단 버튼 (탭 느낌)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("프로필(결과)", use_container_width=True):
        # ✅ 결과 페이지로 바로 이동
            st.session_state.mode = "survey"
            st.session_state.survey_page = "result"
            st.rerun()

    with col2:
        if st.button("설문 다시", use_container_width=True):
            reset_chat()
            go_survey()

    with col3:
        if st.button("대화 초기화", use_container_width=True):
            reset_chat()
            st.rerun()

    with col4:
        end_chat = st.button("종료 요약", use_container_width=True)

   # 현재 상담자 모드 (접기)
    with st.expander("🧭 현재 상담자 모드", expanded=False):
        st.write(
            f"- **톤**: {persona_rule.get('tone','')}\n"
            f"- **목표**: {persona_rule.get('goal','')}"
        )
    # VectorDB/LLM 로드 (chat 모드에서만 시도)
    try:
        stores = load_vectorstores_only()
        counsel_db = stores["counsel_db"]
        risk_db = stores["risk_db"]
        llm = get_llm()
    except Exception as e:
        st.error(f"VectorDB/LLM 로드 실패: {e}")
        st.stop()

    # 메시지 출력
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # 종료 요약
    if end_chat:
        if not st.session_state.messages:
            st.info("아직 대화가 없습니다.")
        else:
            summary = final_summary_fewshot(
                llm=llm,
                history_summary=st.session_state.history_summary,
                risk_mode=bool(st.session_state.get("ever_risk", False)),
            )
            st.subheader("✅ 상담 종료 요약")
            st.text(summary)

    # 입력
    user_text = st.chat_input("지금 어떤 점이 가장 마음에 걸리세요?")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        out = run_turn(
            llm=llm,
            persona_rule=persona_rule,
            counsel_db=counsel_db,
            risk_db=risk_db,
            history_summary=st.session_state.history_summary,
            user_message=user_text,
        )

        st.session_state.history_summary = out["history_summary"]
        st.session_state.messages.append({"role": "assistant", "content": out["assistant_answer"]})
        st.session_state.ever_risk = st.session_state.ever_risk or bool(out.get("risk_mode", False))

        with st.chat_message("assistant"):
            st.write(out["assistant_answer"])

        st.rerun()


# =========================================================
# 9) 라우팅 (survey/chat)
# =========================================================
if "mode" not in st.session_state:
    st.session_state.mode = "survey"

init_survey_state()
init_chat_state()

if st.session_state.mode == "survey":
    if st.session_state.survey_page == "survey":
        render_survey()
    else:
        render_result()
else:
    render_chat()
    
