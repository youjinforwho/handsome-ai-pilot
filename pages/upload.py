import streamlit as st
import os
import time
import uuid
import json
import pickle
import io
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
from brand_data import get_brand_from_filename

# ★ [중요] 구글 신형 SDK (v1.0) 임포트
from google import genai
from google.genai import types

# 구글 드라이브 업로드용 라이브러리
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# 1. 환경변수 로드
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# API 키 확인
if not GOOGLE_API_KEY:
    st.error("구글 API 키가 없습니다. .env 파일을 확인해주세요.")
    st.stop()

if not PINECONE_API_KEY:
    st.error("파인콘 API 키가 없습니다. .env 파일을 확인해주세요.")
    st.stop()

# ★ [중요] 신형 클라이언트 초기화 (GenerativeModel 아님!)
client = genai.Client(api_key=GOOGLE_API_KEY)


# ==========================================
# [기능 0] 사이드바 DB 상태 확인
# ==========================================
with st.sidebar:
    st.header("📊 DB 상태 조회")
    if st.button("내 DB 찔러보기"):
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            st.success("연결 성공!")
            st.write(f"총 데이터: **{stats['total_vector_count']}개**")
            if 'namespaces' in stats:
                st.json(stats['namespaces'])
        except Exception as e:
            st.error(f"연결 실패: {e}")

# ==========================================
# [기능 1] 구글 드라이브 업로드
# ==========================================
def upload_to_google_drive(file_obj, filename):
    FOLDER_ID = os.getenv("FOLDER_ID")

    
    CLIENT_SECRET_FILE = 'client_secret.json'
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    if not os.path.exists(CLIENT_SECRET_FILE):
        st.error(f"구글 OAuth 파일({CLIENT_SECRET_FILE})을 찾을 수 없습니다.")
        return None
    
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': filename, 'parents': [FOLDER_ID]}
        media = MediaIoBaseUpload(file_obj, mimetype=file_obj.type, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()
        return file.get('webViewLink')

    except Exception as e:
        st.error(f"구글 드라이브 업로드 실패: {e}")
        return None

# ==========================================
# [기능 2-1] 이미지의 시각적 느낌(분위기, 무드)을 텍스트로 추출
# ==========================================
# Gemini 공개 API는 이미지를 직접 임베딩하는 멀티모달 embed API를 제공하지 않습니다.
# 대신 generate_content로 이미지를 분석해 "느낌"을 한글 문장으로 뽑고,
# 이 텍스트를 나중에 브랜드/태그와 합쳐 한 번에 임베딩합니다.
# → 결과적으로 한 벡터에 "이미지 느낌 + 메타데이터"가 모두 반영됩니다.
def generate_image_vibe_description(image) -> str:
    """
    이미지를 보고 시각적 분위기, 무드, 색감, 스타일 감성을 2~3문장 한글로 설명합니다.
    반환된 문자열은 검색용 통합 텍스트 앞쪽에 붙여서 임베딩에 사용됩니다.
    """
    prompt = """이 패션/의류 이미지의 시각적 느낌을 설명해주세요.
- 분위기(무드), 색감, 전체적인 스타일 감성, 착용감에 대한 인상을 2~3문장 한글로만 작성해주세요.
- JSON이나 목록 형식 없이, 연속된 문단(플레인 텍스트)으로만 답하세요.
- 다른 설명이나 접두어 없이 바로 본문만 출력하세요."""

    # 태깅과 동일하게 여러 모델명을 순서대로 시도 (API 변경 대비)
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
# [기능 2-2] 이미지 느낌 + 메타데이터 통합 텍스트를 한 번에 벡터화
# ==========================================
# 사용 모델: gemini-embedding-001 (Gemini 공개 API에서 사용 가능한 텍스트 임베딩 모델)
# - 이미지 픽셀을 직접 넣는 멀티모달 임베딩은 Vertex AI 전용이므로, 여기서는
#   "이미지 느낌 설명(vibe) + 브랜드 + 태그"를 하나의 긴 텍스트로 만들어
#   텍스트 임베딩으로 한 벡터에 모두 담습니다.
# - 출력 차원: 1408. Pinecone 인덱스 dimension=1408 필요.
EMBED_DIM = 1408


def get_multimodal_embedding(image, text: str):
    """
    이미지 느낌(vibe) 텍스트와 메타데이터가 이미 합쳐진 text를 1408차원 벡터로 임베딩합니다.
    - image: 호출 규약 유지용(내부에서 사용하지 않음). 실제로는 text만 API에 전달합니다.
    - text: "이미지 분위기 설명 + 브랜드 + 카테고리 + 색상 + 스타일 + ..." 형태의 통합 검색용 텍스트
    """
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBED_DIM  # 1408
            ),
        )

        return response.embeddings[0].values
    except Exception as e:
        print(f"임베딩 생성 실패: {e}")
        return None

# ==========================================
# [기능 3] 태깅 생성 (신형 SDK 사용)
# ==========================================
# [수정된 함수] 모델 이름을 'gemini-2.0-flash'로 변경
def generate_tags(image):
    prompt = """
    Role: You are a Senior Merchandiser (MD) at Handsome with 20 years of experience.
    Task: Analyze the visual elements of the image and extract structured data for search optimization.
    Constraints: Output ONLY JSON. Values in Korean.
    [JSON Structure]
    {
      "cat": "Item Name", "col": ["Color"], "mat": "Material",
      "pat": "Pattern", "sty": "Style", "sea": "Season",
      "neck": "Neckline", "fit": "Fit", "det": ["Detail"]
    }
    """
    # ★ 시도할 모델 목록 (우선순위 순)
    # 하나가 안 되면 다음 걸로 자동으로 넘어갑니다.
    candidate_models = [
        'gemini-2.0-flash',       # 1순위: 표준 별칭
        'gemini-2.5-flash',       # 2순위: 최신 모델
        'gemini-2.5-flash-image', # 3순위: 이미지 처리 모델
        'gemini-2.5-flash-video', # 4순위: 비디오 처리 모델
        'gemini-2.5-flash-audio', # 5순위: 오디오 처리 모델
        'gemini-2.5-flash-text', # 6순위: 텍스트 처리 모델
        'gemini-2.5-flash-text-image', # 7순위: 텍스트와 이미지 처리 모델
        'gemini-2.5-flash-text-video', # 8순위: 텍스트와 비디오 처리 모델
        'gemini-2.5-flash-text-audio', # 9순위: 텍스트와 오디오 처리 모델
    ]

    last_error = ""

    for model_name in candidate_models:
        try:
            # 모델 호출 시도
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image]
            )
            
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            
            # 성공하면 바로 리턴 (루프 종료)
            # print(f"✅ 성공한 모델: {model_name}") # 디버깅용
            return text.strip()
            
        except Exception as e:
            # 실패하면 에러 기록하고 다음 모델로 넘어감
            last_error = str(e)
            continue
    
    # 모든 모델이 실패했을 경우에만 에러 리턴
    return json.dumps({"error": f"모든 모델 시도 실패. 마지막 에러: {last_error}"})

# ==========================================
# [UI] 화면 구성
# ==========================================
st.title("☁️ 한섬 AI 포토 클라우드 (New SDK)")
st.caption("이미지 느낌 + 브랜드·태그 통합 벡터 저장 (Gemini 1408차원)")


if "process_results" not in st.session_state:
    st.session_state.process_results = []

uploaded_files = st.file_uploader(
    "이미지를 드래그하세요", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 10:
    st.warning("⚠️ 파일럿 테스트는 최대 10장까지만 가능합니다.")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    start_btn = st.button("🚀 업로드 및 처리 시작", type="primary", use_container_width=True)
with col2:
    cancel_btn = st.button("🔄 초기화", use_container_width=True)

if cancel_btn:
    st.rerun()

# ==========================================
# [핵심 로직] 실행
# ==========================================
if start_btn and uploaded_files:
    
    # Pinecone 연결
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        st.error(f"Pinecone 연결 실패: {e}")
        st.stop()

    total_files = len(uploaded_files)
    st.divider()
    process_results = []
    
    with st.status(f"⚙️ 데이터 처리 중... (총 {total_files}장)", expanded=True) as status:
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            current_idx = i + 1
            
            # 1. 정보 추출
            original_name = uploaded_file.name
            brand_name = get_brand_from_filename(original_name)
            
            status.write(f"**[{current_idx}/{total_files}]** {original_name} (브랜드: {brand_name})")
            
            # 2. UUID 생성
            file_ext = os.path.splitext(original_name)[1]
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            
            # 3. 구글 드라이브 업로드
            drive_link = upload_to_google_drive(uploaded_file, unique_filename)
            
            # 4. 이미지 태깅 (업로드 파일 스트림이 소모되는 이슈 방지용으로 bytes로 고정)
            image_bytes = uploaded_file.getvalue()
            image_for_ai = Image.open(io.BytesIO(image_bytes))
            json_str = generate_tags(image_for_ai)
            
            try:
                # 5-1. JSON 파싱
                data = json.loads(json_str)
                if "error" in data: raise Exception(data["error"])

                colors = " ".join(data.get('col', [])) if isinstance(data.get('col'), list) else str(data.get('col'))
                if data.get('neck') in ["없음", "None"]: data['neck'] = ""
                # 검색용 메타데이터 텍스트: 브랜드, 카테고리, 색상, 스타일, 소재, 넥라인, 핏, 디테일
                metadata_text = f"{brand_name} {data.get('cat')} {colors} {data.get('sty')} {data.get('mat')} {data.get('neck')} {data.get('fit')} {data.get('det')}"

                # 5-2. 이미지 느낌(분위기, 무드)을 텍스트로 추출 → 메타데이터와 합쳐 한 벡터로 임베딩
                with st.spinner("이미지 느낌 분석 중..."):
                    vibe_text = generate_image_vibe_description(image_for_ai)
                # 통합 검색용 텍스트: [이미지 느낌 설명] + [브랜드·태그 등] → 이 전체를 한 번에 벡터화
                combined_text = f"{vibe_text} {metadata_text}".strip() if vibe_text else metadata_text

                with st.spinner("이미지 느낌 + 메타데이터 통합 임베딩 생성 중 (1408차원)..."):
                    vector_embedding = get_multimodal_embedding(image=image_for_ai, text=combined_text)

                # 5-3. Pinecone 업로드 (벡터 차원 1408. 인덱스 dimension=1408 필요)
                if vector_embedding:
                    try:
                        # 저장되는 벡터: 이미지 분위기 설명(vibe) + 브랜드·카테고리·색상·스타일 등 메타데이터가
                        # 하나의 텍스트로 합쳐진 뒤 gemini-embedding-001 로 1408차원으로 임베딩된 값
                        upsert_response = index.upsert(
                            vectors=[
                                {
                                    "id": unique_filename,
                                    "values": vector_embedding,
                                    "metadata": {
                                        "id": unique_filename,
                                        "original_name": original_name,
                                        "brand": brand_name,
                                        "drive_link": drive_link,
                                        "category": data.get('cat', ''),
                                        "style": data.get('sty', ''),
                                        "vibe": vibe_text,  # ★ [추가] Pinecone에도 느낌 텍스트 저장
                                        "detail_json": json_str
                                    }
                                }
                            ]
                        )
                        
                        # pinecone SDK 버전에 따라 dict 또는 객체로 반환될 수 있음
                        upserted_count = None
                        if isinstance(upsert_response, dict):
                            upserted_count = upsert_response.get("upserted_count", 0)
                        else:
                            upserted_count = getattr(upsert_response, "upserted_count", 0)

                        if upserted_count and upserted_count > 0:
                            process_results.append({
                                "id": unique_filename,
                                "status": "success",
                                "filename": original_name,
                                "brand": brand_name,
                                "image_bytes": image_bytes,
                                "json_data": data,
                                "vibe_text": vibe_text,  # ★ [추가] 결과 화면 출력을 위해 여기에 저장
                                "tags": f"{data.get('cat')} / {data.get('sty')}",
                                "embedding_dim": len(vector_embedding)
                            })
                            status.write(f" └ ✅ 저장 성공! (Count: {upserted_count})")
                        else:
                            # 0개 저장 시 에러 처리
                            raise Exception("Pinecone upsert returned 0 count")

                    except Exception as e:
                        st.error(f"❌ Pinecone 저장 에러: {e}")
                        raise e
                else:
                    raise Exception("임베딩 생성 실패로 Pinecone 업로드 불가")
                
            except Exception as e:
                st.error(f"❌ 실패: {e}")
                process_results.append({
                    "status": "fail",
                    "filename": original_name,
                    "error": str(e)
                })
            
            progress_bar.progress(current_idx / total_files)

            # ★ [필수 추가] 무료 버전 한도(분당 15회)를 지키기 위한 강제 휴식
            if i < total_files - 1:
                with st.spinner("API 과부하 방지를 위해 4초 대기 중..."):
                    time.sleep(4)
        
        status.update(label="🎉 작업 완료!", state="complete", expanded=False)

    # 결과를 세션에 저장 (rerun되어도 하단 결과 유지)
    st.session_state.process_results = process_results

elif not uploaded_files and start_btn:
    st.warning("이미지를 선택해주세요.")

# ==========================================
# [결과] 언제나 하단에 렌더링 (세션에 남아있으면 표시)
# ==========================================
if st.session_state.process_results:
    st.divider()
    st.subheader(f"📸 처리 결과 ({len(st.session_state.process_results)}장)")

    for result in st.session_state.process_results:
        if result.get("status") == "success":
            with st.container(border=True):
                col_thumb, col_info = st.columns([1, 3])
                with col_thumb:
                    st.image(result["image_bytes"], use_container_width=True)
                with col_info:
                    st.markdown(f"#### {result['tags']}")
                    st.caption(
                        f"파일: `{result['filename']}` | 브랜드: **{result['brand']}** | 임베딩: `{result['embedding_dim']}차원`"
                    )
                    # ★ [추가] 이미지 느낌(Vibe) 보여주는 부분
                    st.markdown("**🌊 AI가 분석한 시각적 느낌:**")
                    st.info(result.get("vibe_text", "분석된 내용이 없습니다."))

                    with st.expander("🔍 상세 태그 정보 (JSON)"):
                        st.json(result["json_data"])
        else:
            with st.container(border=True):
                st.error(f"❌ 실패: {result.get('filename','(unknown)')}")
                st.caption(f"사유: {result.get('error','(no error message)')}")