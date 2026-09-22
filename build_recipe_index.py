# 레시피 문서로 FAISS 인덱스 구축
# 실행: python build_recipe_index.py
# 결과: documents.index, documents.json (main.py가 그대로 불러다 씀)

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "documents.index"
DOCS_PATH = "documents.json"
DIMENSION = 768  # ko-sroberta-multitask 출력 차원

def main():
    # 1. 레시피 문서 불러오기
    print("레시피 문서 로딩 중...")
    with open('recipe_documents.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"{len(documents)}개 문서 로드 완료")

    # 2. 임베딩 모델 로딩 & 벡터화
    print("임베딩 모델 로딩 중...")
    embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    print("문서 임베딩 중... (문서 수에 따라 몇 분 소요될 수 있음)")
    #show_progress_bar=True: 진행 상황을 실시간으로 보여줌 (대량 데이터 처리 시 유용)
    vecs = embed_model.encode(
        documents, show_progress_bar=True, batch_size=32
    ).astype('float32')

    # 코사인 유사도를 위한 정규화
    faiss.normalize_L2(vecs)
    print("임베딩 shape:",vecs.shape)    # (문서수, 786)

    # 3. FAISS 인덱스 구축 & 저장
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(vecs)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"완료: {index.ntotal}개 문서가 인덱싱됨")
    print(f"저장 위치: {INDEX_PATH}, {DOCS_PATH}")

    # 4. 간단한 검증 - 실제 검색이 잘 되는지 확인
    print("\n=== 검증: 샘플 검색 ===")
    test_queries = ["매운 음식 추천해줘", "다이어트에 좋은 음식", "간단하게 만들 수 있는 요리"]

    for query in test_queries:
        query_vec = embed_model.encode(query).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, 2)

        print(f"\n질문: {query}")
        for i, s in zip(indices[0], scores[0]):
            # 문서가 길어서 요리명 부분만 잘라서 미리보기
            preview = documents[i][:40]
            print(f"    {s:.3f}: {preview}...")

if __name__ == "__main__":
    main()