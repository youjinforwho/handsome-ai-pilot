"""
자유문장 텍스트 검색 + 이미지 벡터 검색 + 이미지 웹 유사도 검색.
- 탭1: Pinecone 기반 텍스트 검색 (Gemini 1408차원)
- 탭2: Pinecone 기반 이미지 검색 (upload.py와 동일한 vibe+태그 임베딩)
- 탭3: Google Cloud Vision WEB_DETECTION (웹 유사 이미지, DB 미사용)

[정확도 개선 전략 - 성별/연령 하드 필터]
upload.py가 Pinecone에 저장하는 최상위 메타데이터: brand, category, style, vibe, detail_json
→ gen(성별), age(연령대)는 detail_json 내부에만 존재 → Pinecone filter 파라미터 직접 사용 불가

해결책: post-filter 방식
  1. top_k=50으로 넉넉하게 후보 수집
  2. detail_json 파싱 → gen/age 필드 추출
  3. 성별 하드 필터(여성 쿼리면 남성 결과 제거) 적용
  4. 연령대 소프트 필터(점수 조정) 적용
  5. 최종 상위 N개만 표시
"""
import os
import re
import io
import json
import pickle
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from PIL import Image

from google import genai
from google.genai import types
import requests as _requests
from google.auth.transport.requests import Request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

load_dotenv()

_cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _cred_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _cred_path

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "handsome-ai-pilot")

if not GOOGLE_API_KEY:
    st.error("구글 API 키가 없습니다. .env에 GOOGLE_API_KEY를 설정해주세요.")
    st.stop()
if not PINECONE_API_KEY:
    st.error("Pinecone API 키가 없습니다. .env에 PINECONE_API_KEY를 설정해주세요.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)


# ==========================================
# [0] Google Drive 이미지 로딩 (서버사이드 bytes 방식)
# ==========================================
def prefetch_images(results: list):
    file_ids = [drive_link_to_file_id(r["drive_link"]) for r in results if r.get("drive_link")]
    file_ids = [fid for fid in file_ids if fid]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_image_bytes_cached, fid): fid for fid in file_ids}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass


def drive_link_to_file_id(drive_link: Optional[str]) -> Optional[str]:
    if not drive_link:
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", drive_link)
    return m.group(1) if m else None


def _get_bearer_token() -> Optional[str]:
    if not os.path.exists('token.pickle'):
        return None
    try:
        with open('token.pickle', 'rb') as f:
            creds = pickle.load(f)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open('token.pickle', 'wb') as f:
                pickle.dump(creds, f)
        return creds.token if creds and creds.valid else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_image_bytes_cached(file_id: str) -> Optional[bytes]:
    token = _get_bearer_token()
    if token:
        try:
            resp = _requests.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
        except Exception:
            pass

    try:
        resp = _requests.get(
            f"https://lh3.googleusercontent.com/d/{file_id}",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass

    try:
        resp = _requests.get(
            f"https://drive.google.com/thumbnail?id={file_id}&sz=w400",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass

    return None


def render_drive_image(drive_link: Optional[str]):
    file_id = drive_link_to_file_id(drive_link)
    if not file_id:
        st.caption("(이미지 없음)")
        return
    img_bytes = _fetch_image_bytes_cached(file_id)
    if img_bytes:
        try:
            st.image(img_bytes, use_container_width=True)
            return
        except Exception:
            pass
    st.caption("⚠️ 이미지 로딩 실패")
    if drive_link:
        st.markdown(f"[🔗 Drive에서 직접 열기]({drive_link})")


# ==========================================
# [1] 텍스트 임베딩
# ==========================================
def get_text_embedding(text: str):
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=1408),
        )
        return response.embeddings[0].values
    except Exception as e:
        st.error(f"임베딩 생성 실패: {e}")
        return None


# ==========================================
# [2] 자연어 → 구조화 쿼리 (정확도 개선 버전)
# ==========================================
def structure_query_for_search(user_query: str) -> Optional[dict]:
    """
    사용자 자연어를 Gemini로 구조화.
    gen(성별), age(연령대) 필드를 정확히 추출하는 것이 핵심.
    query_text는 DB 저장 형식(vibe + 메타데이터)과 동일한 구조로 생성.
    """
    prompt = """
Role: You are a Senior Fashion Search Expert at Handsome with 20 years of experience.
Task: Convert user's natural language into a structured search query.
Output: ONLY valid JSON. Korean values only. No markdown, no explanation.

[Critical extraction rules]
- gen(성별): MUST be extracted precisely.
  - "남성", "남자", "남" → "남성"
  - "여성", "여자", "여" → "여성"
  - "남녀", "커플", "unisex" → "남녀공용"
  - 언급 없으면 → "" (빈 문자열, 절대 추정하지 말 것)
- age(연령대): "10대", "20대", "30대", "3040", "4050", "전연령" 중에서 추출. 언급 없으면 "".
- cat(카테고리): 코트/자켓/점퍼/가디건/니트/셔츠/블라우스/티셔츠/팬츠/데님/스커트/원피스/정장/스웨터/슈즈/백/액세서리 등. 언급 없으면 "".
- occasion(착용 상황): 소개팅/데이트/출근/결혼식/하객/캐주얼/스포츠 등. 새로운 필드로 추출.

[Color normalization]
검정→블랙, 흰색→화이트, 회색→그레이, 아이보리/오프화이트→아이보리

[Fit normalization]
슬림핏/레귤러핏/루즈핏/오버핏/크롭/롱/와이드/스트레이트 중 1개

[query_text construction rules]
DB에 저장된 벡터는 "이미지 분위기 문장 + 메타데이터 키워드" 형식으로 임베딩됨.
query_text도 반드시 같은 형식으로 작성:
  ① 분위기/무드/상황 묘사 문장 (1~2문장, 자연스러운 한국어)
  ② 키워드 나열: 카테고리 색상 스타일 소재 핏 디테일 착용상황

[query_text examples]
입력: "20대 남성 소개팅 룩 추천해줘"
query_text: "깔끔하고 세련된 무드의 남성 캐주얼 룩이다. 소개팅에 잘 어울리는 단정하면서도 스타일리시한 느낌이다. 남성 캐주얼 소개팅 데이트 클린핏"

입력: "결혼식 갈 때 입을 블랙 트위드 자켓"
query_text: "클래식하고 단정한 포멀 분위기의 고급스러운 룩이다. 하객룩으로 잘 어울리는 세련된 스타일이다. 자켓 블랙 트위드 클래식 포멀 하객룩 결혼식"

입력: "여름 린넨 와이드 팬츠"
query_text: "시원하고 자연스러운 내추럴 무드의 편안한 캐주얼 룩이다. 여름 데일리로 활용하기 좋은 가벼운 느낌이다. 팬츠 린넨 캐주얼 와이드핏 여름"

[Required JSON schema]
{
  "cat": "",
  "col": [],
  "mat": "",
  "pat": "",
  "sty": "",
  "sea": "",
  "neck": "",
  "fit": "",
  "det": [],
  "mon": [],
  "gen": "",
  "age": "",
  "occasion": "",
  "query_text": ""
}

User input: """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt + user_query.strip(),
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        st.error(f"쿼리 구조화 실패: {e}")
        return None


# ==========================================
# [3] 이미지 → 분위기 텍스트
# ==========================================
def generate_image_vibe_description(image) -> str:
    prompt = """이 패션/의류 이미지의 시각적 느낌을 설명해주세요.
- 분위기(무드), 색감, 전체적인 스타일 감성, 착용감에 대한 인상을 2~3문장 한글로만 작성해주세요.
- JSON이나 목록 형식 없이, 연속된 문단(플레인 텍스트)으로만 답하세요.
- 다른 설명이나 접두어 없이 바로 본문만 출력하세요."""

    for model_name in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-preview-05-20"]:
        try:
            response = client.models.generate_content(model=model_name, contents=[prompt, image])
            text = (response.text or "").strip()
            if text:
                return text
        except Exception:
            continue
    return ""


# ==========================================
# [4] 이미지 → 태그 JSON
# ==========================================
def generate_tags(image) -> str:
    prompt = """
    Role: You are a Senior Merchandiser (MD) at Handsome with 20 years of experience.
    Task: Analyze the visual elements of the image and extract structured data for search optimization.
    Constraints: Output ONLY JSON. Values in Korean.
    [JSON Structure]
    {
      "cat": "Item Name", "col": ["Color"], "mat": "Material",
      "pat": "Pattern", "sty": "Style", "sea": "Season",
      "neck": "Neckline", "fit": "Fit", "det": ["Detail"],
      "mon": ["Suitable Month (e.g., 3월, 4월, 10월)"],
      "gen": "Target Gender/Category (e.g., 여성, 남성, 남녀공용, 신발, 가방, 잡화)",
      "age": "Target Age Group (e.g., 20대, 3040, 전연령)"
    }
    """
    last_error = ""
    for model_name in ['gemini-2.0-flash', 'gemini-2.5-flash']:
        try:
            response = client.models.generate_content(model=model_name, contents=[prompt, image])
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            return text.strip()
        except Exception as e:
            last_error = str(e)
            continue
    return json.dumps({"error": f"모든 모델 시도 실패. 마지막 에러: {last_error}"})


# ==========================================
# [5] Pinecone 검색 (후처리 필터용 top_k 확대)
# ==========================================
def pinecone_query(vector, top_k: int = 50) -> list:
    """
    벡터 검색. top_k를 크게 설정해 post-filter 이후에도 충분한 결과 확보.
    detail_json도 함께 파싱해 gen/age 필드를 result에 포함.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        res = index.query(vector=vector, top_k=top_k, include_metadata=True)
    except Exception as e:
        st.error(f"Pinecone 검색 실패: {e}")
        return []

    matches = getattr(res, "matches", None) or res.get("matches", [])
    results = []
    for m in matches:
        meta = (getattr(m, "metadata", None) or m.get("metadata")) or {}
        score = getattr(m, "score", None) or m.get("score") or 0

        # detail_json 파싱해서 gen/age 추출 (post-filter에 사용)
        detail_json_str = meta.get("detail_json", "")
        try:
            detail_obj = json.loads(detail_json_str) if isinstance(detail_json_str, str) else detail_json_str
        except Exception:
            detail_obj = {}

        results.append({
            "score": score,
            "brand": meta.get("brand", ""),
            "category": meta.get("category", ""),
            "style": meta.get("style", ""),
            "drive_link": meta.get("drive_link", ""),
            "original_name": meta.get("original_name", ""),
            "detail_json": detail_json_str,
            # post-filter용 필드
            "gen": detail_obj.get("gen", ""),
            "age": detail_obj.get("age", ""),
        })
    return results


# ==========================================
# [6] 핵심: 구조화 쿼리 기반 후처리 필터 + 정렬
# ==========================================

# [gen 하드 필터 규칙]
# "남성" 쿼리 → "남성", "남녀공용" 허용 / "여성" 완전 제거
# "여성" 쿼리 → "여성", "남녀공용" 허용 / "남성" 완전 제거
GEN_EXCLUDE_MAP = {
    "남성": ["여성"],
    "여성": ["남성"],
    "남녀공용": [],
    "": [],
}

# [age 소프트 필터]
# 일치: ×1.1 보너스 / 불일치: ×0.7 페널티
AGE_GROUPS = {
    "10대": ["10대"],
    "20대": ["20대"],
    "30대": ["3040", "30대"],
    "3040": ["3040", "30대", "40대"],
    "4050": ["4050", "40대", "50대"],
    "전연령": None,  # None = 항상 허용
}

# [카테고리 소프트 필터]
# 일치: ×1.3 보너스 / 불일치: ×0.35 페널티 (연령보다 훨씬 강한 가중치)
#
# 동의어 그룹: DB에 저장된 카테고리 표기가 다양할 수 있으므로
# 같은 의미의 카테고리를 묶어 유연하게 매칭.
CATEGORY_SYNONYM_GROUPS: List[set] = [
    {"자켓", "재킷", "블레이저"},
    {"코트", "오버코트", "트렌치코트", "트렌치"},
    {"팬츠", "바지", "슬랙스", "트라우저"},
    {"데님", "청바지", "진"},
    {"스커트", "치마"},
    {"원피스", "드레스"},
    {"니트", "스웨터", "풀오버"},
    {"가디건"},
    {"셔츠", "남방"},
    {"블라우스", "블라우즈"},
    {"티셔츠", "티", "반팔", "롱슬리브", "탑"},
    {"점퍼", "패딩", "다운", "아우터"},
    {"정장", "수트", "슈트"},
    {"슈즈", "신발", "구두", "스니커즈", "운동화", "로퍼", "부츠", "샌들"},
    {"백", "가방", "파우치", "크로스백", "숄더백", "토트백", "클러치"},
    {"액세서리", "주얼리", "목걸이", "귀걸이", "반지", "벨트", "스카프", "모자"},
]

def _norm_cat(cat: str) -> str:
    """카테고리 소문자·공백 제거 정규화."""
    return cat.strip().lower().replace(" ", "")

def _category_matches(query_cat: str, result_cat: str) -> bool:
    """
    쿼리 카테고리와 결과 카테고리가 같은 동의어 그룹이거나
    부분 문자열 포함 관계이면 True (일치로 판정).
    """
    if not query_cat or not result_cat:
        return False  # 한쪽이 비어있으면 판정 불가 → 중립 처리(호출부에서 분기)

    q = _norm_cat(query_cat)
    r = _norm_cat(result_cat)

    if q == r:
        return True

    # 동의어 그룹 매칭
    for group in CATEGORY_SYNONYM_GROUPS:
        normed = {_norm_cat(c) for c in group}
        if q in normed and r in normed:
            return True

    # 부분 문자열 포함 (예: 쿼리="트렌치" → result="트렌치코트" 허용)
    return q in r or r in q


def apply_post_filters(
    results: list,
    structured: dict,
    display_n: int = 12,
) -> list:
    """
    Pinecone 벡터 검색 결과에 필터·가중치를 적용해 재정렬.

    필터 종류 및 가중치:
      - 성별   : 하드 필터 (불일치 시 완전 제거)
      - 카테고리: 소프트 필터, 가중치 ★★★ (일치 ×1.3 / 불일치 ×0.35)
      - 연령대 : 소프트 필터, 가중치 ★   (일치 ×1.1 / 불일치 ×0.7)
    """
    query_gen = (structured.get("gen") or "").strip()
    query_age = (structured.get("age") or "").strip()
    query_cat = (structured.get("cat") or "").strip()

    exclude_gens = GEN_EXCLUDE_MAP.get(query_gen, [])

    filtered = []
    for r in results:
        result_gen = (r.get("gen") or "").strip()
        result_age = (r.get("age") or "").strip()
        result_cat = (r.get("category") or "").strip()  # Pinecone 최상위 메타데이터에서 바로 사용
        score = r["score"]

        # ── 하드 필터: 성별 (불일치 시 완전 제거) ────
        if result_gen in exclude_gens:
            continue

        # ── 소프트 필터: 카테고리 (가중치 최강) ──────
        # 카테고리가 쿼리에 명시된 경우에만 적용
        if query_cat:
            if not result_cat:
                # DB에 카테고리 정보 없음 → 중립 (페널티 없음)
                pass
            elif _category_matches(query_cat, result_cat):
                # 카테고리 일치 → 강한 보너스
                score = min(score * 1.3, 1.0)
            else:
                # 카테고리 불일치 → 강한 페널티 (결과에서 제거하지는 않음)
                score = score * 0.35

        # ── 소프트 필터: 연령대 (카테고리보다 약한 가중치) ──
        if query_age and query_age != "전연령":
            allowed_ages = AGE_GROUPS.get(query_age, [query_age])
            if allowed_ages is not None:
                if result_age == "전연령" or not result_age:
                    pass  # 연령 정보 없음 → 중립
                elif result_age in allowed_ages:
                    score = min(score * 1.1, 1.0)   # 일치 보너스
                else:
                    score = score * 0.7             # 불일치 페널티

        # ── 유사도 임계값 필터: 조정된 score가 0.5 미만이면 제거 ──
        if score < 0.5:
            continue

        # ── 유사도 임계값: 조정된 score 0.5 미만이면 제거 ──
        if score < 0.5:
            continue

        filtered.append({**r, "score": score})

    # 조정된 점수로 재정렬 후 상위 N개 반환
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:display_n]


# ==========================================
# [7] 결과 필터 + 렌더링 공통 함수
# ==========================================
def render_search_results(results: list, filter_key_prefix: str):
    with st.spinner("🖼️ 이미지 로딩 중..."):
        prefetch_images(results)

    all_brands = ["전체"] + sorted({r["brand"] for r in results if r["brand"]})
    all_cats   = ["전체"] + sorted({r["category"] for r in results if r["category"]})
    all_styles = ["전체"] + sorted({r["style"] for r in results if r["style"]})

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        filter_brand    = st.selectbox("브랜드",   all_brands, key=f"{filter_key_prefix}_brand")
    with filter_col2:
        filter_category = st.selectbox("카테고리", all_cats,   key=f"{filter_key_prefix}_category")
    with filter_col3:
        filter_style    = st.selectbox("스타일",   all_styles, key=f"{filter_key_prefix}_style")

    filtered = results
    if filter_brand    and filter_brand    != "전체":
        filtered = [r for r in filtered if r["brand"]    == filter_brand]
    if filter_category and filter_category != "전체":
        filtered = [r for r in filtered if r["category"] == filter_category]
    if filter_style    and filter_style    != "전체":
        filtered = [r for r in filtered if r["style"]    == filter_style]

    st.subheader(f"📸 검색 결과 ({len(filtered)}건)")

    for row_start in range(0, len(filtered), 4):
        row_items = filtered[row_start: row_start + 4]
        cols = st.columns(4)
        for idx, r in enumerate(row_items):
            with cols[idx]:
                with st.container(border=True):
                    render_drive_image(r["drive_link"])
                    st.caption(f"**유사도:** {r['score']:.3f}")
                    st.caption(f"**파일:** {r['original_name']}")
                    st.caption(
                        f"**브랜드:** {r['brand']} | **카테고리:** {r['category']} | **스타일:** {r['style']}"
                    )
                    if r["drive_link"]:
                        st.markdown(f"[🔗 원본 Drive 파일 보기]({r['drive_link']})", unsafe_allow_html=False)
                    detail = r.get("detail_json")
                    if detail:
                        try:
                            detail_obj = json.loads(detail) if isinstance(detail, str) else detail
                        except Exception:
                            detail_obj = {"raw": detail}
                        with st.expander("상세 정보 (JSON)"):
                            st.json(detail_obj)


# ==========================================
# [8] Vision API WEB_DETECTION
# ==========================================
def run_web_detection(image_bytes: bytes) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    if not VISION_AVAILABLE:
        return [], [], []
    vision_client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = vision_client.web_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)
    web = response.web_detection
    labels = [lb.label for lb in (web.best_guess_labels or [])]
    similar_urls = [img.url for img in (web.visually_similar_images or []) if getattr(img, "url", None)]
    pages = []
    for p in web.pages_with_matching_images or []:
        url   = getattr(p, "url",        None) or ""
        title = getattr(p, "page_title", None) or ""
        if url:
            pages.append((url, title))
    return labels, similar_urls, pages


def url_to_domain(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


# ==========================================
# [UI]
# ==========================================
st.set_page_config(page_title="이미지 검색", layout="wide")
st.title("🔍 이미지 검색")
st.markdown("""
<style>
[data-testid="stImage"] img {
  width: 100% !important;
  height: auto !important;
  display: block !important;
}
</style>
""", unsafe_allow_html=True)

# session_state 초기화
for _key, _default in [
    ("search_results",     []),
    ("structured_query",   {}),
    ("img_search_results", []),
    ("img_search_vibe",    ""),
    ("img_search_tags",    {}),
    ("tab1_reset_counter", 0),
    ("tab2_reset_counter", 0),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

tab1, tab2, tab3 = st.tabs(["📝 텍스트로 검색", "🖼️ 이미지로 검색", "🌐 이미지로 웹 유사도 검색"])


# ──────────────────────────────────────────
# 탭1: 텍스트로 검색
# ──────────────────────────────────────────
with tab1:
    st.caption("원하는 스타일을 문장으로 입력하면 유사 이미지를 찾아줍니다. (Gemini 1408차원 + Pinecone)")

    query_input = st.text_input(
        "검색어",
        placeholder="예: 20대 남성 소개팅 룩 추천해줘",
        label_visibility="collapsed",
        key=f"text_query_input_{st.session_state.tab1_reset_counter}",
    )

    btn_col1, btn_col2 = st.columns([5, 1])
    with btn_col1:
        search_clicked = st.button("🔍 검색", type="primary", use_container_width=True, key="text_search_btn")
    with btn_col2:
        reset_clicked  = st.button("🔄 초기화", use_container_width=True, key="text_reset_btn")

    if reset_clicked:
        st.session_state.search_results   = []
        st.session_state.structured_query = {}
        st.session_state.tab1_reset_counter += 1
        st.rerun()

    if search_clicked and query_input.strip():
        with st.spinner("쿼리 구조화 및 검색 중..."):
            structured = structure_query_for_search(query_input.strip())
            if not structured:
                st.stop()

            query_text = structured.get("query_text") or query_input.strip()

            # 구조화된 쿼리 + 적용된 필터 정보 표시
            with st.expander("📋 구조화된 쿼리 및 적용 필터"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**추출된 조건**")
                    filter_info = {
                        "성별(하드필터)": structured.get("gen") or "미지정(전체)",
                        "카테고리(소프트★★★)": structured.get("cat") or "미지정(전체)",
                        "연령대(소프트★)": structured.get("age") or "미지정(전체)",
                        "착용상황": structured.get("occasion") or "미지정",
                    }
                    for k, v in filter_info.items():
                        st.markdown(f"- **{k}**: {v}")
                with col_b:
                    st.markdown("**전체 JSON**")
                    st.json(structured)

            vector = get_text_embedding(query_text)
            if not vector:
                st.stop()

            # top_k=50으로 넉넉하게 수집 후 post-filter
            raw_results = pinecone_query(vector, top_k=50)
            filtered_results = apply_post_filters(raw_results, structured, display_n=12)

            st.session_state.search_results   = filtered_results
            st.session_state.structured_query = structured

            # 필터 적용 결과 안내
            gen = structured.get("gen")
            age = structured.get("age")
            cat = structured.get("cat")
            if gen or age or cat:
                filter_desc = []
                if gen:
                    filter_desc.append(f"성별: **{gen}**")
                if cat:
                    filter_desc.append(f"카테고리: **{cat}** (강)")
                if age:
                    filter_desc.append(f"연령대: **{age}** (약)")
                st.info(f"🔍 필터 적용됨 — {' | '.join(filter_desc)} | 후보 {len(raw_results)}건 → 최종 {len(filtered_results)}건")

    if st.session_state.search_results:
        render_search_results(st.session_state.search_results, filter_key_prefix="tab1")
    elif search_clicked and not query_input.strip():
        st.warning("검색어를 입력해주세요.")
    elif search_clicked and st.session_state.search_results == []:
        st.info("유사도 0.5 이상의 결과가 없습니다. 검색어를 바꿔 다시 시도해보세요.")
    elif search_clicked and not st.session_state.search_results:
        st.info("유사도 0.5 이상의 결과가 없습니다. 검색어를 바꿔 다시 시도해보세요.")


# ──────────────────────────────────────────
# 탭2: 이미지로 검색
# ──────────────────────────────────────────
with tab2:
    st.caption(
        "옷 이미지를 올리면 AI가 분위기와 태그를 분석해 유사한 아이템을 찾아줍니다. "
        "(upload.py와 동일한 Gemini 1408차원 + Pinecone)"
    )

    img_search_file = st.file_uploader(
        "이미지 업로드",
        type=["png", "jpg", "jpeg"],
        key=f"img_search_file_{st.session_state.tab2_reset_counter}",
    )

    btn_col1, btn_col2 = st.columns([5, 1])
    with btn_col1:
        img_search_clicked = st.button(
            "🔍 이미지로 검색",
            type="primary",
            use_container_width=True,
            key="img_search_btn",
            disabled=(img_search_file is None),
        )
    with btn_col2:
        img_reset_clicked = st.button("🔄 초기화", use_container_width=True, key="img_reset_btn")

    if img_reset_clicked:
        st.session_state.img_search_results = []
        st.session_state.img_search_vibe    = ""
        st.session_state.img_search_tags    = {}
        st.session_state.tab2_reset_counter += 1
        st.rerun()

    if img_search_file:
        preview_col, _ = st.columns([1, 3])
        with preview_col:
            st.image(img_search_file, caption="업로드된 이미지", use_container_width=True)

    if img_search_clicked and img_search_file:
        image_bytes  = img_search_file.getvalue()
        image_for_ai = Image.open(io.BytesIO(image_bytes))

        with st.spinner("🌊 이미지 분위기 분석 중..."):
            vibe_text = generate_image_vibe_description(image_for_ai)

        with st.spinner("🏷️ 패션 태그 추출 중..."):
            json_str  = generate_tags(image_for_ai)
            try:
                tags_data = json.loads(json_str)
                if "error" in tags_data:
                    st.warning(f"태그 추출 부분 실패: {tags_data['error']} — vibe만으로 검색합니다.")
                    tags_data = {}
            except Exception:
                tags_data = {}

        colors = " ".join(tags_data.get('col', [])) if isinstance(tags_data.get('col'), list) else str(tags_data.get('col', ''))
        if tags_data.get('neck') in ["없음", "None"]: tags_data['neck'] = ""
        metadata_text = f"{tags_data.get('cat','')} {colors} {tags_data.get('sty','')} {tags_data.get('mat','')} {tags_data.get('neck','')} {tags_data.get('fit','')} {tags_data.get('det','')}"
        combined_text = f"{vibe_text} {metadata_text}".strip() if vibe_text else metadata_text

        with st.spinner("🔎 유사 아이템 검색 중..."):
            vector = get_text_embedding(combined_text)
            if vector:
                # 이미지 검색은 이미지 자체가 성별 컨텍스트를 담고 있으므로
                # 추출된 태그의 gen으로 post-filter 적용
                raw_results = pinecone_query(vector, top_k=50)
                img_structured = {"gen": tags_data.get("gen", ""), "age": tags_data.get("age", ""), "cat": tags_data.get("cat", "")}
                filtered_results = apply_post_filters(raw_results, img_structured, display_n=12)

                st.session_state.img_search_results = filtered_results
                st.session_state.img_search_vibe    = vibe_text
                st.session_state.img_search_tags    = tags_data
            else:
                st.error("임베딩 생성에 실패했습니다.")

    if st.session_state.img_search_vibe or st.session_state.img_search_tags:
        with st.expander("🔍 AI 분석 결과 보기", expanded=False):
            if st.session_state.img_search_vibe:
                st.markdown("**🌊 이미지 분위기**")
                st.info(st.session_state.img_search_vibe)
            if st.session_state.img_search_tags:
                st.markdown("**🏷️ 추출된 태그**")
                st.json(st.session_state.img_search_tags)

    if st.session_state.img_search_results:
        render_search_results(st.session_state.img_search_results, filter_key_prefix="tab2")
    elif img_search_clicked and not img_search_file:
        st.warning("이미지를 업로드해주세요.")
    elif img_search_clicked and st.session_state.img_search_results == []:
        st.info("유사도 0.5 이상의 결과가 없습니다. 다른 이미지로 다시 시도해보세요.")
    elif img_search_clicked and not st.session_state.img_search_results:
        st.info("유사도 0.5 이상의 결과가 없습니다. 다른 이미지로 다시 시도해보세요.")


# ──────────────────────────────────────────
# 탭3: 이미지로 웹 유사도 검색
# ──────────────────────────────────────────
with tab3:
    st.caption("옷 이미지를 올리면 웹에서 유사 이미지·출처 페이지를 찾습니다. (Google Cloud Vision, DB 미사용)")
    if not VISION_AVAILABLE:
        st.warning(
            "Google Cloud Vision 패키지가 필요합니다. "
            "터미널에서 `pip install google-cloud-vision` 후 재실행해주세요."
        )
    img_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"], key="web_search_image")
    if img_file:
        image_bytes = img_file.getvalue()

        labels, similar_urls, pages = [], [], []
        if st.button("🖼️ 웹 유사도 검색", type="primary", key="web_search_btn"):
            if not VISION_AVAILABLE:
                st.error("Google Cloud Vision 패키지를 설치해주세요.")
            else:
                with st.spinner("Vision API WEB_DETECTION 호출 중..."):
                    try:
                        labels, similar_urls, pages = run_web_detection(image_bytes)
                    except Exception as e:
                        st.error(f"Vision API 오류: {e}")

        st.subheader("🏷️ 구글 추정 키워드 (best_guess_labels)")
        if labels:
            st.info(" | ".join(labels))
        else:
            st.caption("(없음)")

        st.subheader("🖼️ 유사 이미지 (visually_similar_images)")
        if similar_urls:
            for row_start in range(0, len(similar_urls), 4):
                row_urls = similar_urls[row_start: row_start + 4]
                cols = st.columns(4)
                for idx, url in enumerate(row_urls):
                    with cols[idx]:
                        with st.container(border=True):
                            st.image(url, use_container_width=True)
                            st.caption(f"[이미지 링크]({url})")
        else:
            st.caption("(유사 이미지 없음)")

        st.subheader("🔗 출처 페이지 (pages_with_matching_images)")
        if pages:
            for url, title in pages:
                domain = url_to_domain(url)
                st.markdown(f"- **{domain}** — [{title or url}]({url})")
        else:
            st.caption("(매칭 페이지 없음)")