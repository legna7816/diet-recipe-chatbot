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
app = FastAPI(title="RAG API", description="검색 기반 질의응답 API")

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

DEFAULT_DOCUMENTS = [
    "타이타닉은 1912년 4월 15일 빙산과 충돌해 침몰한 영국의 여객선이다.",
    "파이썬은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어이다.",
    "BERT는 구글이 2018년에 발표한 자연어처리 모델이다.",
    "김치는 발효 채소를 이용한 한국의 전통 음식이다.",
    "RAG는 검색과 생성을 결합한 자연어처리 기법이다.",
    "딥러닝은 인공신경망을 여러 층으로 쌓아 학습하는 머신러닝의 한 분야이다.",
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
        print("저장된 인덱스 불러오는 중...")
        idx = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        print(f"인덱스 로드 완료 (문서 {len(docs)}개)")
        return idx, docs

    print("세 인덱스 생성 중...")
    idx = faiss.IndexFlatIP(DIMENSION)
    idx.add(embed_and_normalize(DEFAULT_DOCUMENTS))
    docs = DEFAULT_DOCUMENTS.copy()
    save_index(idx, docs)
    print(f"인덱스 생성 완료 (문서 {len(docs)}개)")
    return idx, docs

def save_index(idx, docs):
    """인덱스와 문서 목록을 파일로 저장 (서버 재시작 시 재사용)"""
    faiss.write_index(idx, INDEX_PATH)
    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

index, documents = build_index()

# 3. 요청/응답 형식 정의 (Pydantic)
class QueryRequest(BaseModel):
    question: str
    top_k: int = 2

class QueryResponse(BaseModel):
    question: str
    retrieved_docs: list[str]
    answer: str

class SearchResponse(BaseModel):
    question: str
    results: list[dict]

class AddDocumentRequest(BaseModel):
    documents: list[str]

class AddDocumentResponse(BaseModel):
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
    prompt = f"""당신은 레시피 추천 챗봇입니다. 아래 [레시피 목록]에 있는 요리만 추천하세요. 목록에 없는 요리나 정보는 절대 언급하지 마세요. 영양 성분에 대한 추가 설명은 하지 말고, 요리명과 이유만 간단히 답하세요.

[레시피 목록]
{context}

사용자 질문: {query}
답변 (요리명과 매운 이유만 1~2줄로):"""
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
    """서버 상태 확인 (헬스 체크)"""
    return {
        "status": "running",
        "device": str(device),
        "indexed_documents": index.ntotal
    }

@app.get("/documents")
def list_documents():
    """등록된 문서 목록 조회"""
    return {"count": len(documents), "documents": documents}

@app.post("/documents", response_model=AddDocumentResponse)
def add_documents(request: AddDocumentRequest):
    """새 문서를 인덱스에 추가 (서버 재시작 없이 지식 베이스 확장)"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="문서가 비어있습니다.")
    # 임베딩 -> 정규화 -> 인덱스에 추가
    index.add(embed_and_normalize(request.documents))
    # 인덱스 번호와 순서를 맞추기 위해 리스트에도 동일하게 추가
    documents.extend(request.documents)
    save_index(index, documents)

    return AddDocumentResponse(added=len(request.documents), total=index.ntotal)

@app.post("/search", response_model=SearchResponse)
def search_only(request: QueryRequest):
    """검색만 수행 (생성 없이 빠르게 확인용)"""
    results = search_with_scores(request.question, top_k=request.top_k)
    return SearchResponse(question=request.question, results=results)

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """RAG 전체 파이프라인: 검색 + 답변 생성"""
    results = search_with_scores(request.question, top_k=request.top_k)
    retrieved = [r["document"] for r in results]
    context = "\n".join(retrieved)
    answer = generate_answer(request.question, context)

    return QueryResponse(
        question=request.question,
        retrieved_docs=retrieved,
        answer=answer
    )