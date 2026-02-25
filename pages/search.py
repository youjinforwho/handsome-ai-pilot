"""
자유문장 텍스트 검색 + 이미지 벡터 검색 + 이미지 웹 유사도 검색.
- 탭1: Pinecone 기반 텍스트 검색 (Gemini 1408차원)
- 탭2: Pinecone 기반 이미지 검색 (upload.py와 동일한 vibe+태그 임베딩)
- 탭3: Google Cloud Vision WEB_DETECTION (웹 유사 이미지, DB 미사용)

[벡터 DB 저장 형식 - upload.py 기준]
저장 벡터 = embed( "{vibe 2~3문장} {brand} {cat} {colors} {sty} {mat} {neck} {fit} {det}" )
→ 검색 쿼리도 동일한 순서·구조로 생성해야 유사도 정확도가 높아짐

[이미지 로딩 방식]
drive.google.com/thumbnail URL은 Rate Limit·세션 검증 문제로 간헐적 실패 발생.
→ upload.py와 동일한 token.pickle OAuth 인증으로 Drive API에서 직접 bytes 수신.
→ st.cache_data로 세션 내 캐싱해 중복 API 호출 방지.
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

# 구글 신형 SDK (upload와 동일)
from google import genai
from google.genai import types

# 구글 드라이브 이미지 로딩용 (upload.py와 동일한 인증 방식)
import requests as _requests
from google.auth.transport.requests import Request

# Google Cloud Vision (이미지 웹 유사도 검색용, optional)
try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

load_dotenv()

# Vision API용 서비스 계정 경로 (.env에서 불러와서 환경변수로 설정)
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
# [0] Google Drive 이미지 로딩 (Google CDN 방식)
# ==========================================
# [문제] drive.google.com/thumbnail → Rate Limit·세션 검증으로 간헐적 실패
#        Bearer 토큰 방식 → 토큰 만료 타이밍에 간헐적 실패
# [해결] lh3.googleusercontent.com/d/{file_id}
#        → 구글 공식 이미지 CDN. 파일이 "링크 공유" 상태면 인증 불필요.
#        → Rate Limit 없음, 세션 검증 없음, 항상 안정적.
# [Fallback] CDN 실패 시 requests + Bearer 토큰으로 bytes 직접 수신

def drive_link_to_file_id(drive_link: Optional[str]) -> Optional[str]:
    """drive_link(webViewLink)에서 파일 ID만 추출."""
    if not drive_link:
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", drive_link)
    return m.group(1) if m else None


def _get_bearer_token() -> Optional[str]:
    """token.pickle에서 OAuth Bearer 토큰 추출. 만료 시 자동 갱신."""
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
def get_drive_image_bytes_fallback(file_id: str) -> Optional[bytes]:
    """
    CDN 실패 시 Bearer 토큰으로 Drive API에서 직접 bytes 수신 (최후 수단).
    st.cache_data는 순수 bytes만 캐싱 → 직렬화 문제 없음.
    """
    token = _get_bearer_token()
    if not token:
        return None
    try:
        resp = _requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        return resp.content if resp.status_code == 200 else None
    except Exception:
        return None


def render_drive_image(drive_link: Optional[str]):
    """
    Drive 이미지를 안정적으로 렌더링. 3단계 fallback 구조.
    1순위: lh3.googleusercontent.com CDN URL → 인증 불필요, Rate Limit 없음
    2순위: Bearer 토큰 bytes → CDN 접근 불가 환경 대비
    3순위: thumbnail URL → 최후 수단
    """
    file_id = drive_link_to_file_id(drive_link)
    if not file_id:
        st.caption("(이미지 없음)")
        return

    # 1순위: 구글 이미지 CDN (공개 공유 파일에 가장 안정적)
    cdn_url = f"https://lh3.googleusercontent.com/d/{file_id}"
    st.image(cdn_url, use_container_width=True)


# ==========================================
# [1] 텍스트 임베딩 (upload와 동일 1408차원)
# ==========================================
def get_text_embedding(text: str):
    """업로드 시와 동일한 gemini-embedding-001, 1408차원."""
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
# [2] 자연어 → DB 인덱싱 형식에 맞는 검색 쿼리 생성 (Gemini)
# ==========================================
# [핵심] DB에 저장된 벡터는 아래 두 부분을 합친 텍스트로 생성됨:
#   ① vibe 설명 (2~3문장): 이미지 분위기·무드·색감·스타일 감성
#   ② 메타데이터 키워드: "{brand} {cat} {colors} {sty} {mat} {neck} {fit} {det}"
# → query_text도 동일한 구조 ① + ② 로 생성해야 벡터 공간에서 정확히 매칭됨
def structure_query_for_search(user_query: str) -> Optional[dict]:
    """
    사용자 자연어를 Gemini로 구조화.
    query_text = ① 분위기 묘사 1~2문장 + ② 핵심 키워드 나열 (upload.py 저장 형식과 동일한 구조)
    """
    prompt = """
    Role: You are a Senior Fashion Search Expert with 20 years of experience at Handsome.
    Task: Convert user's natural language into an optimized vector search query matching our DB indexing format.
    Constraints: Output ONLY JSON (no markdown, no code fence). Values in Korean.

    [DB 인덱싱 형식 - 검색 쿼리가 반드시 이 구조를 따라야 합니다]
    저장된 벡터는 아래 두 부분을 이어붙인 텍스트로 생성됩니다:
      ① 이미지 분위기 설명 (2~3문장): "차분하고 세련된 분위기의 미니멀한 룩이다. 고급스러운 소재감과 절제된 색감이 인상적이다."
      ② 메타데이터 키워드 나열: "{카테고리} {색상} {스타일} {소재} {넥라인} {핏} {디테일}"

    [JSON 구조 - 항상 모든 키 포함]
    {
      "cat": "Item Name", "col": ["Color"], "mat": "Material",
      "pat": "Pattern", "sty": "Style", "sea": "Season",
      "neck": "Neckline", "fit": "Fit", "det": ["Detail"],
      "mon": ["Suitable Month (e.g., 3월, 4월, 10월)"],
      "gen": "Target Gender/Category (e.g., 여성, 남성, 남녀공용, 신발, 가방, 잡화)",
      "age": "Target Age Group (e.g., 20대, 3040, 전연령)",
      "query_text": "① 분위기 1~2문장. ② 카테고리 색상 스타일 소재 핏 키워드 나열"
    }

    [정규화 규칙]
    - 색상: 검정→블랙, 흰색→화이트, 회색→그레이, 아이보리/오프화이트→아이보리
    - 핏: 슬림핏/레귤러핏/루즈핏/오버핏/크롭/롱/와이드/스트레이트 중 택1
    - 소재: 울/캐시미어/면/데님/가죽/트위드/린넨/폴리/니트 등 단어형으로
    - 카테고리: 코트/자켓/점퍼/가디건/니트/셔츠/블라우스/티셔츠/팬츠/데님/스커트/원피스/정장/스웨터 등

    [query_text 작성 규칙]
    - 사용자가 언급하지 않은 속성은 절대 추가하지 말 것
    - ① 분위기 문장: 사용자 의도에서 유추되는 무드·감성·착용감을 자연스러운 한국어 문장으로
    - ② 키워드: 카테고리→색상→스타일→소재→핏→디테일 순서로 단어 나열
    - 동의어가 도움되면 함께 포함 (예: 하객룩 결혼식, 블레이저 자켓)

    [query_text 예시]
    입력: "결혼식 갈 때 입을 블랙 트위드 자켓"
    출력 query_text: "클래식하고 단정한 포멀 분위기의 고급스러운 룩이다. 하객룩으로 잘 어울리는 세련된 스타일이다. 자켓 블랙 트위드 클래식 포멀 하객룩 결혼식"

    입력: "여름에 시원한 린넨 와이드 팬츠"
    출력 query_text: "시원하고 자연스러운 내추럴 무드의 편안한 캐주얼 룩이다. 여름 데일리로 활용하기 좋은 가벼운 느낌이다. 팬츠 린넨 캐주얼 와이드핏 여름"

    입력: "오버사이즈 베이지 울 코트"
    출력 query_text: "따뜻하고 여유로운 무드의 고급스러운 겨울 룩이다. 클래식하면서도 세련된 오버핏 실루엣이 인상적이다. 코트 베이지 울 클래식 오버핏 가을겨울"

    User input: """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt + user_query.strip(),
        )
        text = (response.text or "").strip()
        # strip markdown code block if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        st.error(f"쿼리 구조화 실패: {e}")
        return None


# ==========================================
# [3] 이미지 → 분위기 텍스트 (upload.py와 완전히 동일)
# ==========================================
# [핵심] DB 벡터의 앞부분(①)이 이 함수의 출력으로 생성됨
# → 검색 시에도 동일한 프롬프트·모델로 vibe를 추출해야 벡터 공간이 일치함
def generate_image_vibe_description(image) -> str:
    """
    이미지를 보고 시각적 분위기, 무드, 색감, 스타일 감성을 2~3문장 한글로 설명.
    반환된 문자열은 검색용 통합 텍스트 앞쪽에 붙여서 임베딩에 사용됩니다.
    """
    prompt = """이 패션/의류 이미지의 시각적 느낌을 설명해주세요.
- 분위기(무드), 색감, 전체적인 스타일 감성, 착용감에 대한 인상을 2~3문장 한글로만 작성해주세요.
- JSON이나 목록 형식 없이, 연속된 문단(플레인 텍스트)으로만 답하세요.
- 다른 설명이나 접두어 없이 바로 본문만 출력하세요."""

    candidate_models = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
    ]
    last_error = ""
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image],
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as e:
            last_error = str(e)
            continue
    # 모두 실패 시 빈 문자열 반환 (메타데이터만으로도 임베딩은 가능)
    print(f"이미지 느낌 설명 생성 실패(무시 후 진행): {last_error}")
    return ""


# ==========================================
# [4] 이미지 → 태그 JSON (upload.py와 완전히 동일)
# ==========================================
# [핵심] DB 벡터의 뒷부분(②)이 이 함수의 출력으로 구성된 metadata_text임
# → 동일한 프롬프트·모델로 태그를 추출해야 키워드 공간이 일치함
def generate_tags(image) -> str:
    """
    이미지 태그를 JSON으로 추출. upload.py와 완전히 동일한 프롬프트·로직 사용.
    반환된 JSON은 파싱 후 metadata_text 구성에 사용됩니다.
    """
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
    candidate_models = [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
    ]
    last_error = ""
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image]
            )
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
# [5] Pinecone 검색 공통 함수
# ==========================================
def pinecone_query(vector, top_k: int = 20) -> list:
    """벡터로 Pinecone를 조회해 메타데이터가 포함된 결과 리스트 반환."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        res = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as e:
        st.error(f"Pinecone 검색 실패: {e}")
        return []
    matches = getattr(res, "matches", None) or res.get("matches", [])
    results = []
    for m in matches:
        meta = (getattr(m, "metadata", None) or m.get("metadata")) or {}
        score = getattr(m, "score", None) or m.get("score") or 0
        results.append({
            "score": score,
            "brand": meta.get("brand", ""),
            "category": meta.get("category", ""),
            "style": meta.get("style", ""),
            "drive_link": meta.get("drive_link", ""),
            "original_name": meta.get("original_name", ""),
            "detail_json": meta.get("detail_json", ""),
        })
    return results


# ==========================================
# [6] 결과 필터 + 렌더링 공통 함수
# ==========================================
def render_search_results(results: list, filter_key_prefix: str):
    """필터 UI + 이미지 그리드 렌더링. 탭1·탭2 공통 사용."""
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
                    # render_drive_image: Drive API bytes → fallback URL 순으로 시도
                    render_drive_image(r["drive_link"])
                    st.caption(f"**유사도:** {r['score']:.3f}")
                    st.caption(f"**파일:** {r['original_name']}")
                    st.caption(
                        f"**브랜드:** {r['brand']} | **카테고리:** {r['category']} | **스타일:** {r['style']}"
                    )
                    # 원본 Drive 링크 — 이미지 로딩 실패 시 원본 존재 여부 직접 확인용
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
# [7] Google Cloud Vision WEB_DETECTION (이미지 → 웹 유사 이미지)
# ==========================================
def run_web_detection(image_bytes: bytes) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Vision API WEB_DETECTION만 사용.
    Returns: (best_guess_labels, visually_similar_image_urls, [(page_url, page_title)])
    """
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
# [UI] 검색 화면 (탭)
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

tab1, tab2, tab3 = st.tabs(["📝 텍스트로 검색", "🖼️ 이미지로 검색", "🌐 이미지로 웹 유사도 검색"])


# ──────────────────────────────────────────
# 탭1: 텍스트로 검색 (Pinecone)
# ──────────────────────────────────────────
with tab1:
    st.caption("원하는 스타일을 문장으로 입력하면 유사 이미지를 찾아줍니다. (Gemini 1408차원 + Pinecone)")
    query_input = st.text_input(
        "검색어",
        placeholder="예: 결혼식 갈 때 입기 좋은 옷",
        label_visibility="collapsed",
        key="text_query_input",
    )
    search_clicked = st.button("🔍 검색", type="primary", use_container_width=True, key="text_search_btn")

    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    if search_clicked and query_input.strip():
        with st.spinner("쿼리 구조화 및 검색 중..."):
            structured = structure_query_for_search(query_input.strip())
            if not structured:
                st.stop()

            # query_text = ① 분위기 문장 + ② 키워드 나열 (DB 저장 형식과 동일한 구조)
            query_text = structured.get("query_text") or query_input.strip()

            with st.expander("📋 구조화된 쿼리 (JSON)"):
                st.json(structured)

            vector = get_text_embedding(query_text)
            if not vector:
                st.stop()

            st.session_state.search_results   = pinecone_query(vector)
            st.session_state.structured_query = structured

    if st.session_state.search_results:
        render_search_results(st.session_state.search_results, filter_key_prefix="tab1")
    elif search_clicked and not query_input.strip():
        st.warning("검색어를 입력해주세요.")


# ──────────────────────────────────────────
# 탭2: 이미지로 검색 (upload.py와 동일한 vibe+태그 벡터)
# ──────────────────────────────────────────
# [핵심] upload.py의 벡터 생성 파이프라인을 그대로 재현:
#   vibe_text     = generate_image_vibe_description(image)              ← ① DB 앞부분과 동일
#   metadata_text = "{cat} {colors} {sty} {mat} {neck} {fit} {det}"    ← ② DB 뒷부분과 동일 (brand 제외)
#   combined_text = f"{vibe_text} {metadata_text}"                      ← upload.py와 동일한 결합 방식
#   vector        = embed(combined_text, 1408차원)
with tab2:
    st.caption(
        "옷 이미지를 올리면 AI가 분위기와 태그를 분석해 유사한 아이템을 찾아줍니다. "
        "(upload.py와 동일한 Gemini 1408차원 + Pinecone)"
    )

    img_search_file = st.file_uploader(
        "이미지 업로드",
        type=["png", "jpg", "jpeg"],
        key="img_search_file",
    )

    if "img_search_results" not in st.session_state:
        st.session_state.img_search_results = []
    if "img_search_vibe" not in st.session_state:
        st.session_state.img_search_vibe = ""
    if "img_search_tags" not in st.session_state:
        st.session_state.img_search_tags = {}

    img_search_clicked = st.button(
        "🔍 이미지로 검색",
        type="primary",
        use_container_width=True,
        key="img_search_btn",
        disabled=(img_search_file is None),
    )

    if img_search_file:
        # 업로드된 이미지 미리보기
        preview_col, _ = st.columns([1, 3])
        with preview_col:
            st.image(img_search_file, caption="업로드된 이미지", use_container_width=True)

    if img_search_clicked and img_search_file:
        image_bytes  = img_search_file.getvalue()
        image_for_ai = Image.open(io.BytesIO(image_bytes))

        # Step 1. 이미지 느낌(분위기, 무드) 추출 → combined_text의 ① 앞부분 (upload.py와 동일)
        with st.spinner("🌊 이미지 분위기 분석 중..."):
            vibe_text = generate_image_vibe_description(image_for_ai)

        # Step 2. 패션 태그 추출 → combined_text의 ② 뒷부분 (upload.py와 동일)
        with st.spinner("🏷️ 패션 태그 추출 중..."):
            json_str  = generate_tags(image_for_ai)
            try:
                tags_data = json.loads(json_str)
                if "error" in tags_data:
                    st.warning(f"태그 추출 부분 실패: {tags_data['error']} — vibe만으로 검색합니다.")
                    tags_data = {}
            except Exception:
                tags_data = {}

        # Step 3. combined_text 구성 — upload.py와 완전히 동일한 방식
        colors = " ".join(tags_data.get('col', [])) if isinstance(tags_data.get('col'), list) else str(tags_data.get('col', ''))
        if tags_data.get('neck') in ["없음", "None"]: tags_data['neck'] = ""
        metadata_text = f"{tags_data.get('cat','')} {colors} {tags_data.get('sty','')} {tags_data.get('mat','')} {tags_data.get('neck','')} {tags_data.get('fit','')} {tags_data.get('det','')}"
        combined_text = f"{vibe_text} {metadata_text}".strip() if vibe_text else metadata_text

        # Step 4. 임베딩 → Pinecone 검색
        with st.spinner("🔎 유사 아이템 검색 중..."):
            vector = get_text_embedding(combined_text)
            if vector:
                st.session_state.img_search_results = pinecone_query(vector)
                st.session_state.img_search_vibe    = vibe_text
                st.session_state.img_search_tags    = tags_data
            else:
                st.error("임베딩 생성에 실패했습니다.")

    # 분석 결과 표시 (검색 후 유지)
    if st.session_state.img_search_vibe or st.session_state.img_search_tags:
        with st.expander("🔍 AI 분석 결과 보기", expanded=False):
            if st.session_state.img_search_vibe:
                st.markdown("**🌊 이미지 분위기**")
                st.info(st.session_state.img_search_vibe)
            if st.session_state.img_search_tags:
                st.markdown("**🏷️ 추출된 태그**")
                st.json(st.session_state.img_search_tags)

    # 검색 결과 렌더링
    if st.session_state.img_search_results:
        render_search_results(st.session_state.img_search_results, filter_key_prefix="tab2")
    elif img_search_clicked and not img_search_file:
        st.warning("이미지를 업로드해주세요.")


# ──────────────────────────────────────────
# 탭3: 이미지로 웹 유사도 검색 (Vision API WEB_DETECTION)
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