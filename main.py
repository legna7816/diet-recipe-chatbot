"""
다이어트 레시피 챗봇 API (대화 히스토리 + 세션 관리 v2: 이전 레시피 유지 방식)
실행: uvicorn main:app --reload
테스트: http://127.0.0.1:8000/docs
"""

import os
import json
import uuid
import torch
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. 앱 초기화 & 모델 로딩
# ============================================================
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
    torch_dtype=torch.float32,
)
gen_model.to(device)
gen_model.eval()
print("모델 로딩 완료")


# ============================================================
# 2. FAISS 인덱스 & 레시피 문서 저장소
# ============================================================
INDEX_PATH = "documents.index"
DOCS_PATH = "documents.json"
DIMENSION = 768

DEFAULT_DOCUMENTS = [
    "김치찌개는 국&찌개 종류의 요리로, 김치와 돼지고기를 넣고 끓이는 대표적인 한식이다.",
]


def embed_and_normalize(texts):
    """텍스트를 임베딩하고 정규화 (정규화해야 내적 = 코사인 유사도)"""
    if isinstance(texts, str):
        texts = [texts]
    vecs = embed_model.encode(texts).astype('float32')
    vecs = np.atleast_2d(vecs)
    faiss.normalize_L2(vecs)
    return vecs


def build_index():
    """저장된 레시피 인덱스가 있으면 불러오고, 없으면 기본값으로 생성"""
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
    """인덱스와 레시피 목록을 파일로 저장 (서버 재시작 시 재사용)"""
    faiss.write_index(idx, INDEX_PATH)
    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


index, documents = build_index()


# ============================================================
# 3. 세션 & 대화 히스토리 (v2: 메모리 저장)
# ============================================================
# 구조: { session_id: {"history": [...], "last_recipes": [...]} }
#
# v1은 "이전 질문을 검색어에 합치는" 방식이었으나 실패했음:
#   "매운 음식 추천해줘 그거 칼로리는 얼마야?" -> '칼로리'에 끌려 엉뚱한 레시피 검색됨
# v2는 직전 턴에서 쓴 레시피를 세션에 보관했다가 다음 턴 맥락으로 이어붙임
sessions: dict[str, dict] = {}

MAX_TURNS = 3           # 기억할 대화 턴 수 (질문+답변 한 쌍 = 1턴)
MAX_CONTEXT_RECIPES = 4  # 프롬프트에 넣을 레시피 최대 개수 (길어지면 느려지고 품질 저하)


def get_or_create_session(session_id: str | None) -> str:
    """session_id가 없으면 새로 발급, 처음 보는 id면 빈 세션으로 생성"""
    if not session_id:
        session_id = str(uuid.uuid4())
    sessions.setdefault(session_id, {"history": [], "last_recipes": []})
    return session_id


def trim_history(history: list[dict]):
    """최근 MAX_TURNS 턴만 남기고 오래된 대화는 삭제 (제자리 수정)"""
    max_messages = MAX_TURNS * 2
    if len(history) > max_messages:
        del history[:-max_messages]


def merge_recipes(new_recipes: list[str], previous_recipes: list[str]) -> list[str]:
    """
    이번 검색 결과 + 직전 턴에서 쓴 레시피를 합침 (중복 제거, 개수 제한)
    새 결과를 앞에 두되, 직전 레시피도 남겨 "그거"가 가리키는 대상을 유지
    """
    merged = []
    for recipe in previous_recipes + new_recipes:
        if recipe not in merged:
            merged.append(recipe)
    return merged[:MAX_CONTEXT_RECIPES]


# ============================================================
# 4. 요청/응답 형식
# ============================================================
class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None   # 없으면 서버가 새로 발급
    top_k: int = 2

class ChatResponse(BaseModel):
    session_id: str
    question: str
    retrieved_recipes: list[str]    # 이번 검색 결과
    context_recipes: list[str]      # 실제로 프롬프트에 들어간 레시피 (이전 턴 포함)
    answer: str

class SearchRequest(BaseModel):
    question: str
    top_k: int = 2

class SearchResponse(BaseModel):
    question: str
    results: list[dict]

class AddRecipeRequest(BaseModel):
    documents: list[str]

class AddRecipeResponse(BaseModel):
    added: int
    total: int

class HistoryResponse(BaseModel):
    session_id: str
    turns: int
    history: list[dict]


# ============================================================
# 5. RAG 로직
# ============================================================
# 규칙을 짧고 단정적으로 씀
# 지시가 길고 복잡하면 작은 모델이 일부를 무시하는 경향이 있음
SYSTEM_PROMPT = """당신은 다이어트 레시피 추천 챗봇입니다. 다음 규칙을 반드시 지키세요.
1. [레시피 목록]에 있는 요리명만 사용합니다. 목록에 없는 요리는 절대 쓰지 않습니다.
2. 요리는 최대 2개까지만 언급합니다. 목록을 길게 나열하지 않습니다.
3. 칼로리 등 숫자는 [레시피 목록]에 적힌 값만 씁니다. 추측하지 않습니다.
4. [레시피 목록]은 앞쪽일수록 이전 대화에서 언급된 요리이고, 뒤쪽일수록 방금 새로 찾은 요리입니다. "그거"는 이전 대화에서 언급된, 목록 앞쪽의 요리를 가리킵니다.
5. 한국어로만 답합니다. 다른 언어를 섞지 않습니다.
6. 2~3문장으로 짧게 답합니다."""


def search_with_scores(query, top_k=2):
    """FAISS 인덱스로 유사 레시피 검색"""
    query_vec = embed_and_normalize(query)
    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results = []
    for i, s in zip(indices[0], scores[0]):
        if i == -1:
            continue
        results.append({"recipe": documents[i], "score": float(s)})
    return results


def generate_answer(question, context, history):
    """
    히스토리를 포함해 답변 생성
    messages 구성: [system] + [이전 대화들] + [이번 질문 + 레시피 목록]
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"[레시피 목록]\n{context}\n\n질문: {question}"
    })

    text = gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gen_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=180,   # 길면 목록을 지어내며 늘어지므로 제한
            temperature=0.3,      # 낮출수록 지시를 충실히 따름 (창의성 < 정확성)
            do_sample=True
        )
    return gen_tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()


# ============================================================
# 6. API 엔드포인트
# ============================================================

@app.get("/")
def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "device": str(device),
        "recipe_count": index.ntotal,
        "active_sessions": len(sessions)
    }


@app.get("/recipes")
def list_recipes():
    """등록된 레시피 목록 조회"""
    return {"count": len(documents), "recipes": documents}


@app.post("/recipes", response_model=AddRecipeResponse)
def add_recipes(request: AddRecipeRequest):
    """새 레시피를 인덱스에 추가 (서버 재시작 없이 레시피 DB 확장)"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="레시피가 비어있습니다.")

    index.add(embed_and_normalize(request.documents))
    documents.extend(request.documents)
    save_index(index, documents)

    return AddRecipeResponse(added=len(request.documents), total=index.ntotal)


@app.post("/search", response_model=SearchResponse)
def search_only(request: SearchRequest):
    """레시피 검색만 수행 (생성 없이 빠른 확인용, 히스토리 미사용)"""
    results = search_with_scores(request.question, top_k=request.top_k)
    return SearchResponse(question=request.question, results=results)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """챗봇 대화: 검색 + 이전 레시피 유지 + 답변 생성 + 히스토리 저장"""
    session_id = get_or_create_session(request.session_id)
    session = sessions[session_id]
    history = session["history"]

    # 1) 현재 질문으로 검색 (질문을 합치지 않음 - v1에서 검색이 엉뚱해졌던 원인)
    results = search_with_scores(request.question, top_k=request.top_k)
    retrieved = [r["recipe"] for r in results]

    # 2) 직전 턴 레시피와 합쳐서 맥락 유지
    context_recipes = merge_recipes(retrieved, session["last_recipes"])
    context = "\n".join(context_recipes)

    # 3) 히스토리 + 레시피 목록으로 답변 생성
    answer = generate_answer(request.question, context, history)

    # 4) 이번 턴 저장 (레시피 목록은 빼고 순수 질문/답변만 히스토리에)
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})
    trim_history(history)
    session["last_recipes"] = context_recipes

    return ChatResponse(
        session_id=session_id,
        question=request.question,
        retrieved_recipes=retrieved,
        context_recipes=context_recipes,
        answer=answer
    )


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str):
    """세션의 대화 기록 조회"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="존재하지 않는 세션입니다.")
    history = sessions[session_id]["history"]
    return HistoryResponse(
        session_id=session_id,
        turns=len(history) // 2,
        history=history
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """세션 대화 초기화"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="존재하지 않는 세션입니다.")
    del sessions[session_id]
    return {"deleted": session_id}