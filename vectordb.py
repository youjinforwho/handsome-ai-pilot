import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 2. .env 파일에서 API 키 불러오기
load_dotenv()

# 3. DB가 저장될 폴더 이름 지정
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "handsome_fashion_images"

def get_vector_db():
    # Google Gemini 임베딩 모델 사용 (공식 권장 모델명: gemini-embedding-001)
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
    
    return vectordb

if __name__ == "__main__":
    db = get_vector_db()
    print("✅ Google Gemini 임베딩 모델(gemini-embedding-001) 연결 완료!")