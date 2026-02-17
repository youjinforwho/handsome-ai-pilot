"""
자유문장 텍스트 검색 + 이미지 웹 유사도 검색.
- 탭1: Pinecone 기반 텍스트 검색 (Gemini 768차원)
- 탭2: Google Cloud Vision WEB_DETECTION (웹 유사 이미지, DB 미사용)
"""
import os
import re
import json
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

# 구글 신형 SDK (upload와 동일)
from google import genai
from google.genai import types

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
# [1] 텍스트 임베딩 (upload와 동일 768차원, 재사용)
# ==========================================
# ※ SigLIP 등 다른 임베딩은 차원/공간이 달라 Pinecone 저장 벡터와 비교 불가 → 추후 옵션으로만 고려
def get_text_embedding(text: str):
    """업로드 시와 동일한 gemini-embedding-001, 768차원."""
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return response.embeddings[0].values
    except Exception as e:
        st.error(f"임베딩 생성 실패: {e}")
        return None


# ==========================================
# [2] 자연어 → 검색용 구조화 JSON (Gemini)
# ==========================================
def structure_query_for_search(user_query: str) -> Optional[dict]:
    """
    사용자 자연어를 Gemini로 구조화.
    출력: category, style, season, color, material, fit, occasion, query_text (임베딩용, 한국어 유지)
    """
    prompt = """You are a fashion search expert. Given the user's natural language search intent, output ONLY a single JSON object (no markdown, no code fence). Use Korean for all values when possible.
Required keys: "category", "style", "season", "color", "material", "fit", "occasion", "query_text".
- query_text: One concise Korean sentence that describes the search intent for vector embedding (e.g. "결혼식 갈 때 입기 좋은 정장 옷"). This will be used for semantic search.
- Fill other keys with relevant values from the user intent; use empty string "" if not specified.
Example output: {"category": "정장", "style": "클래식", "season": "", "color": "", "material": "", "fit": "", "occasion": "결혼식", "query_text": "결혼식 갈 때 입기 좋은 정장"}
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
# [3] Drive 링크 → 썸네일/이미지 URL
# ==========================================
def drive_link_to_image_url(drive_link: Optional[str], thumbnail_size: int = 300) -> Optional[str]:
    """drive_link(webViewLink)에서 파일 ID를 추출해 이미지 표시용 URL로 변환."""
    if not drive_link:
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", drive_link)
    if not m:
        return drive_link
    file_id = m.group(1)
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w{thumbnail_size}"


# ==========================================
# [4] Google Cloud Vision WEB_DETECTION (이미지 → 웹 유사 이미지)
# ==========================================
def run_web_detection(image_bytes: bytes) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Vision API WEB_DETECTION만 사용.
    Returns: (best_guess_labels, visually_similar_image_urls, [(page_url, page_title)])
    """
    if not VISION_AVAILABLE:
        return [], [], []
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.web_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)
    web = response.web_detection
    labels = [lb.label for lb in (web.best_guess_labels or [])]
    similar_urls = [img.url for img in (web.visually_similar_images or []) if getattr(img, "url", None)]
    pages = []
    for p in web.pages_with_matching_images or []:
        url = getattr(p, "url", None) or ""
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
/* st.image가 카드/컬럼 안에서 가끔 intrinsic width로 잡히는 문제 방지 */
[data-testid="stImage"] img {
  width: 100% !important;
  height: auto !important;
  display: block !important;
}
</style>
""", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📝 텍스트로 검색", "🖼️ 이미지로 웹 유사도 검색"])

# ---------- 탭1: 텍스트로 검색 (기존 Pinecone) ----------
with tab1:
    st.caption("원하는 스타일을 문장으로 입력하면 유사 이미지를 찾아줍니다. (Gemini 768차원 + Pinecone)")
    query_input = st.text_input(
        "검색어",
        placeholder='예: 결혼식 갈 때 입기 좋은 옷',
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
            query_text = structured.get("query_text") or query_input.strip()
            with st.expander("📋 구조화된 쿼리 (JSON)"):
                st.json(structured)
            vector = get_text_embedding(query_text)
            if not vector:
                st.stop()
            try:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                res = index.query(
                    vector=vector,
                    top_k=20,
                    include_metadata=True,
                )
            except Exception as e:
                st.error(f"Pinecone 검색 실패: {e}")
                st.stop()
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
            st.session_state.search_results = results
            st.session_state.structured_query = structured

    if st.session_state.search_results:
        all_brands = ["전체"] + sorted({r["brand"] for r in st.session_state.search_results if r["brand"]})
        all_cats = ["전체"] + sorted({r["category"] for r in st.session_state.search_results if r["category"]})
        all_styles = ["전체"] + sorted({r["style"] for r in st.session_state.search_results if r["style"]})
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            filter_brand = st.selectbox("브랜드", all_brands, key="filter_brand")
        with filter_col2:
            filter_category = st.selectbox("카테고리", all_cats, key="filter_category")
        with filter_col3:
            filter_style = st.selectbox("스타일", all_styles, key="filter_style")
        filtered = st.session_state.search_results
        if filter_brand and filter_brand != "전체":
            filtered = [r for r in filtered if r["brand"] == filter_brand]
        if filter_category and filter_category != "전체":
            filtered = [r for r in filtered if r["category"] == filter_category]
        if filter_style and filter_style != "전체":
            filtered = [r for r in filtered if r["style"] == filter_style]
        st.subheader(f"📸 검색 결과 ({len(filtered)}건)")
        for row_start in range(0, len(filtered), 4):
            row_items = filtered[row_start : row_start + 4]
            cols = st.columns(4)
            for idx, r in enumerate(row_items):
                with cols[idx]:
                    with st.container(border=True):
                        img_url = drive_link_to_image_url(r["drive_link"])
                        if img_url:
                            st.image(img_url, use_container_width=True)
                        else:
                            st.caption("(이미지 없음)")
                        st.caption(f"**유사도:** {r['score']:.3f}")
                        st.caption(f"**파일:** {r['original_name']}")
                        st.caption(f"**브랜드:** {r['brand']} | **카테고리:** {r['category']} | **스타일:** {r['style']}")
                        detail = r.get("detail_json")
                        if detail:
                            try:
                                detail_obj = json.loads(detail) if isinstance(detail, str) else detail
                            except Exception:
                                detail_obj = {"raw": detail}
                            with st.expander("상세 정보 (JSON)"):
                                st.json(detail_obj)
    elif search_clicked and not query_input.strip():
        st.warning("검색어를 입력해주세요.")

# ---------- 탭2: 이미지로 웹 유사도 검색 (Vision API WEB_DETECTION) ----------
with tab2:
    st.caption("옷 이미지를 올리면 웹에서 유사 이미지·출처 페이지를 찾습니다. (Google Cloud Vision, DB 미사용)")
    if not VISION_AVAILABLE:
        st.warning("Google Cloud Vision 패키지가 필요합니다. 터미널에서 `pip install google-cloud-vision` 후 재실행해주세요.")
    img_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"], key="web_search_image")
    if img_file:
        image_bytes = img_file.getvalue()
        st.image(image_bytes, caption="업로드한 이미지", use_container_width=True)
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
                row_urls = similar_urls[row_start : row_start + 4]
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
