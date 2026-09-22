# 식약처 레시피 데이터를 RAG 문서로 변환
# 실행: python build_recipe_documents.py
# 결과: recipe_documents.json (RAG API에 넣을 문서 리스트)

import requests
import json
import time

SERVICE_KEY = "a005b424b1be4eaaa1d5"
BASE_URL = f"http://openapi.foodsafetykorea.go.kr/api/{SERVICE_KEY}/COOKRCP01/json"

# 1. 레시피 가져오기 (전체 또는 일부)
def fetch_recipes(start, end):
    url = f"{BASE_URL}/{start}/{end}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['COOKRCP01']['row']

def fetch_all_recipes(batch_size=100, limit=1156):
    # 전체 1156개를 한 번에 가져오면 응답이 크고 느릴 수 있어
    # batch_size 단위로 나눠서 가져옴 (limit으로 실습용 개수 제한)
    all_recipes = []
    start = 1
    while start <= limit:
        end = min(start + batch_size - 1, limit)
        print(f"가져오는 중: {start}-{end}")
        recipes = fetch_recipes(start, end)
        all_recipes.extend(recipes)
        start = end + 1
        time.sleep(0.3)    # API 서버에 부담 주지 않기 위한 대기
    return all_recipes

# 2. 한국어 조사 처리 함수
def get_josa(word, josa_type='은는'):
    # 한글 단어의 마지막 글자 받침 유무로 알맞은 조사를 반환
    # josa_type:: '은는', '이가', '을를', '과와' 중 선택
    if not word:
        return ''
    last_char = word[-1]

    # 한글이 아니면 (숫자, 영문 등) 기본값 반환
    if not ('가' <= last_char <= '힣'):
        return josa_type[0]

    # 한글 유니코드 구조: (초성*21 + 중성)*28 + 종성 + 0xAC00
    # 종성이 0이면 받침 없음
    code = ord(last_char) - 0xAC00
    has_batchim = (code % 28) != 0

    josa_map = {
        '은는': ('은', '는'),
        '이가': ('이', '가'),
        '을를': ('을', '를'),
        '과와': ('과', '와'),
    }
    with_batchim, without_batchim = josa_map[josa_type]
    return with_batchim if has_batchim else without_batchim

# 3. 레시피 하나를 RAG용 문서(자연어 문단)로 변환
def recipe_to_document(recipe):
    name = recipe.get('RCP_NM', '').strip()
    josa1 = get_josa(name, '은는')

    # 재료: RCP_PARTS_DTLS 앞부분에 요리명이 중복으로 들어있어 제거
    ingredients = recipe.get('RCP_PARTS_DTLS', '').strip()
    if ingredients.startswith(name):
        ingredients = ingredients[len(name):].strip()
    ingredients = ingredients.replace('\n', ', ') # 줄바꿈을 콤마로 정리

    category = recipe.get('RCP_PAT2', '')    # 요리 분류 (반찬, 국&찌개 등)
    cooking_way = recipe.get('RCP_WAY2', '') # 조리 방법 (볶기, 찌기 등)

    # 조리 단계: MANUAL01 ~ MANUAL20 중 값이 있는 것만 순서대로 모으기
    steps = []
    for i in range(1, 21):
        key = f'MANUAL{i:02d}'    # 01, 02, ... , 20 형태로 포맷
        step_text = recipe.get(key, '').strip()
        if step_text:
            steps.append(step_text)
    steps_text = ' '.join(steps)

    # 영양 정보
    calories = recipe.get('INFO_ENG', '')

    # 최종 문서: 검색이 잘 되도록 핵심 정보를 자연어 문장으로 구성
    document = (
        f"{name}{josa1} {category} 종류의 요리로, 조리 방법은 {cooking_way}이다."
        f"재료는 {ingredients}이다."
        f"칼로리는 {calories}kcal이다."
        f"조리 순서: {steps_text}"
    )
    return document

# 4. 전체 실행
if __name__ == "__main__":
    print("레시피 데이터 가져오는 중...")

    # 실습 단계라 300개만 *전체는 1156개, limit 조절 가능
    recipes = fetch_all_recipes(batch_size=100, limit=1156)
    print(f"총 {len(recipes)}개 레시피 수집 완료")

    print("문서로 변환 중...")
    documents = [recipe_to_document(r) for r in recipes]

    # 확인용 샘플 출력
    print("\n=== 샘플 문서 ===")
    print(documents[0])

    # 파일로 저장 -> 다음 단계(FAISS 인덱싱)에서 사용
    with open('recipe_documents.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"\nrecipe_documents.json 저장 완료 ({len(documents)}개 문서)")