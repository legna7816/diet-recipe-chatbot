# 다이어트 레시피 챗봇 API
# 실행: uvicorn main:app --reload
# 테스트: http://127.0.0.1:8000/docs

import os
import json
import torch
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 앱 초기화 & 모델 로딩
# 모델 로딩은 서버 시작 시 한 번만 실행됨
# (요청마다 로딩 시 매번 수십 초가 걸려 서비스 X)
app = FastAPI(
    title="Diet Recipe Chatbot API",
    description="식약처 레시피 데이터 기반 다이어트 레시피 추천 챗봇"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

print("임베딩 모델 로딩 중...")
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

print("생성 모델 로딩 중...")

gen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name,
    torch_dtype=torch.float32,  # CPU는 float16 지원이 불안정하므로 float32 사용
)
gen_model.to(device)
gen_model.eval()
print("모델 로딩 완료")

# 2. FAISS 인덱스 & 문서 저장소
# 인덱스는 벡터만 저장하고 원본 텍스트는 모름
# -> documents 리스트와 인덱스 번호를 항상 같은 순서로 유지해야 함
INDEX_PATH = "documents.index"
DOCS_PATH = "documents.json"

DIMENSION = 768    # ko-sroberta-multitask의 출력 차원

# 기본값: 인덱스 파일이 없을 때만 사용됨
# 실제로는 build_recipe_index.py로 만든 레시피 인덱스를 불러와서 씀
DEFAULT_DOCUMENTS = [
    "김치찌개는 국&찌개 종류의 요리로, 김치와 돼지고기를 넣고 끓이는 대표적인 한식이다.",
]

def embed_and_normalize(texts):
    """텍스트를 임베딩하고 정규화 (정규화해야 내적 = 코사인 유사도)"""
    if isinstance(texts, str):
        texts = [texts]    # 문자열 하나면 리스트로 감싸기
    vecs = embed_model.encode(texts).astype('float32')
    vecs = np.atleast_2d(vecs)
    faiss.normalize_L2(vecs)
    return vecs

def build_index():
    """저장된 인덱스가 있으면 불러오고, 없으면 새로 생성"""
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print("저장된 레시피 인덱스 불러오는 중...")
        idx = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        print(f"레시피 인덱스 로드 완료 (레시피 {len(docs)}개)")
        return idx, docs

    print("레시피 인덱스가 없어 기본값으로 생성합니다.")
    print("실제 레시피 데이터를 쓰려면 build_recipe_documents.py -> build_recipe_index.py 순서로 먼저 실행하세요.")
    idx = faiss.IndexFlatIP(DIMENSION)
    idx.add(embed_and_normalize(DEFAULT_DOCUMENTS))
    docs = DEFAULT_DOCUMENTS.copy()
    save_index(idx, docs)
    return idx, docs

def save_index(idx, docs):
    """인덱스와 문서 목록을 파일로 저장 (서버 재시작 시 재사용)"""
    faiss.write_index(idx, INDEX_PATH)
    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

index, documents = build_index()

# 3. 요청/응답 형식 정의 (Pydantic)
class ChatRequest(BaseModel):
    question: str
    top_k: int = 2

class ChatResponse(BaseModel):
    question: str
    retrieved_docs: list[str]
    answer: str

class SearchResponse(BaseModel):
    question: str
    results: list[dict]

class AddRecipeRequest(BaseModel):
    documents: list[str]

class AddRecipeResponse(BaseModel):
    added: int
    total: int

# 4. RAG 로직
def search_with_scores(query, top_k=2):
    """FAISS 인덱스로 유사 문서 검색"""
    query_vec = embed_and_normalize(query)
    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results = []
    for i, s in zip(indices[0], scores[0]):
        if i == -1:    # 결과가 부족할 때 FAISS는 -1을 반환함
            continue
        results.append({"document": documents[i], "score": float(s)})
    return results

def generate_answer(query, context):
    """레시피 목록을 참고해 사용자 질문에 답변 생성"""
    prompt = f"""당신은 다이어트 레시피 추천 챗봇입니다. 아래 [레시피 목록]에 있는 요리만 추천하세요. 목록에 없는 요리나 정보는 절대 언급하지 마세요. 영양 성분에 대한 추가 설명은 하지 말고, 요리명과 이유만 간단히 답하세요.

[레시피 목록]
{context}

사용자 질문: {query}
답변 (요리명과 이유만 1~2줄로):"""
    messages = [{"role": "user", "content": prompt}]
    text = gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gen_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            do_sample=True
        )
    return gen_tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()

# 5. API 엔드포인트
@app.get("/")
def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "device": str(device),
        "indexed_documents": index.ntotal
    }

@app.get("/recipes")
def list_documents():
    """등록된 fptlvl 목록 조회"""
    return {"count": len(documents), "documents": documents}

@app.post("/recipes", response_model=AddRecipeResponse)
def add_documents(request: AddRecipeRequest):
    """새 레시피를 인덱스에 추가 (서버 재시작 없이 레시피 DB 확장)"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="레시피가 비어있습니다.")
    # 임베딩 -> 정규화 -> 인덱스에 추가
    index.add(embed_and_normalize(request.documents))
    # 인덱스 번호와 순서를 맞추기 위해 리스트에도 동일하게 추가
    documents.extend(request.documents)
    save_index(index, documents)

    return AddRecipeResponse(added=len(request.documents), total=index.ntotal)

@app.post("/search", response_model=SearchResponse)
def search_only(request: ChatRequest):
    """레시피 검색만 수행 (생성 없이 빠르게 확인용)"""
    results = search_with_scores(request.question, top_k=request.top_k)
    return SearchResponse(question=request.question, results=results)

@app.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest):
    """챗봇 대화: 검색 + 답변 생성"""
    results = search_with_scores(request.question, top_k=request.top_k)
    retrieved = [r["recipe"] for r in results]
    context = "\n".join(retrieved)
    answer = generate_answer(request.question, context)

    return ChatResponse(
        question=request.question,
        retrieved_docs=retrieved,
        answer=answer
    )