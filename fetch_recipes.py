# 식약처 조리 식품 레시피 DB API 테스트
# 목적: 데이터 구조 확인 -> RAG 문서로 변환
# 실행: python fetch_recipes.py

import requests
import json

# 1. API 키 설정
# 주의: 실제 서비스에서는 이 키를 코드에 직접 쓰지 않고
# 환경 변수(os.environ)나 .env 파일로 관리해야 함
SERVICE_KEY = "a005b424b1be4eaaa1d5"

# API URL 형식: /서비스키/서비스명/요청타입/시작인덱스/종료인덱스
# COOKRCP01: 조리식품 레시피 DB 서비스명
BASE_URL = f"http://openapi.foodsafetykorea.go.kr/api/{SERVICE_KEY}/COOKRCP01/json"

# 2. 소량 테스트 (5개만 가져와서 구조 확인)
def fetch_recipes(start=1, end=5):
    url = f"{BASE_URL}/{start}/{end}"
    response = requests.get(url)
    response.raise_for_status()    # 요청 실패 시 에러 발생시킴
    return response.json()

print("API 테스트 중...")
data = fetch_recipes(1, 5)

# 전체 구조 확인 
# print(json.dumps(data, ensure_ascii=False, indent=2)[:2000]) # 앞부분만 출력

# 첫 전째 레시피의 모든 키, 이름만 확인 (메뉴명 필드 찾기용)
first_recipe = data['COOKRCP01']['row'][0]
for key in first_recipe.keys():
    print(key, ":", str(first_recipe[key])[:30])

# 3. TODO
# 3-1. 최상위 키
#   -> COOKRCP01
#
# 3-2. 실제 레시피 리스트가 어느 키 안에 있는 지 확인
#   -> row
#
# 3-3. 레시피 하나에 어떤 필드들이 있는지 (메뉴명, 재료, 조리법 필드명 확인)
#   -> 필드	                   용도
#   RCP_NM	                 요리명
#   RCP_PARTS_DTLS           재료 (요리명이 앞에 중복으로 붙어있음, 정리 필요)
#   RCP_WAY2	             조리 방법 (찌기 등)
#   RCP_PAT2	             요리 분류 (반찬 등)
#   MANUAL01~20	             조리 단계 (빈 값 많음, 필터링 필요)
#   INFO_ENG/CAR/PRO/FAT/NA  영양정보
#   HASH_TAG	             해시태그
#   RCP_NA_TIP	             영양 팁