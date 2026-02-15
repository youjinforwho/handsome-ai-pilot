# brand_data.py
# 한섬 브랜드 코드 매핑 정보

BRAND_MAPPING = {
    "TM": "TIME",
    "MN": "MINE",
    "SY": "SYSTEM",
    "SJ": "SJSJ",
    "TH": "TIME HOMME",
    "OR": "오에라",
    "AD": "앤드뮐미스터",
    "AM": "아스페시남성",
    "AW": "아스페시여성",
    "AN": "아뇨나",
    "OB": "Obzee",
    "LC": "랑방 컬렉션",
    "CM": "더캐시미어"    
}

def get_brand_from_filename(filename):
    """
    파일명을 받아서 브랜드 이름을 반환하는 함수
    예: 'TM_24SS_Coat.jpg' -> 'TIME'
    예: 'Unknown.jpg' -> '기타(Etc)'
    """
    if not filename or len(filename) < 2:
        return "기타(Etc)"
    
    # 앞 2글자를 대문자로 추출
    code = filename[:2].upper()
    
    # 매핑된 이름이 있으면 반환, 없으면 '기타' 반환
    return BRAND_MAPPING.get(code, "기타(Etc)")