"""
다이어트 레시피 챗봇 API (대화 히스토리 + 세션 관리 v3: MySQL 저장)
실행 전: schema.sql 실행, .env 설정
실행: uvicorn main:app --reload
테스트: http://127.0.0.1:8000/docs
"""

import os
import json
import uuid
import torch
import numpy as np
import faiss
import pymysql
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()  # .env 파일의 값을 환경변수로 불러옴

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
# 3. DB 저장소 (세션 & 대화 히스토리, v3: MySQL)
# ============================================================
"""
v2의 sessoins dict를 대체, 서버를 재시작해도 대화가 유지됨
구조: chat_sessions(세션 1개) 1:N chat_messages(메시지 여러 개)

SQL의 %s는 JDBC PreparedStatement의 ?와 같은 역할
-> 값을 문자열로 이어붙이지 않고 드라이버가 완전하게 바인딩 (SQL 인젝션 방지)
"""
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "diet_recipe_chatbot"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,  # 결과를 dic로 받음 (row["role"]  형태)
    "autocommit": False                         # 쓰기 작업은 직접 commit (트랜잭션 제어)
}

MAX_TURNS = 3            # 프롬프트에 넣을 최근 대화 턴 수 (저장은 전부, 읽을 때만 제한)
MAX_CONTEXT_RECIPES = 4  # 프롬프트에 넣을 레시피 최대 개수

def get_connection():
    """요청마다 새 연결 생성 (FastAPI 스레드마다 독립된 연결 사용)"""
    return pymysql.connect(**DB_CONFIG)

def check_db_connection():
    """
    서버 시작 시 DB 연결 확인
    문제가 있으면 첫 요청 때가 아니라 시작할 때 바로 알 수 있게
    """
    try:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM chat_sessions")
            print(f"DB 연결 완료 (저장된 세션 {cur.fetchone()['cnt']}개)")
    except pymysql.MySQLError as e:
        raise RuntimeError(
            f"DB 연결 실패: {e}\n.env 설정과 schema.sql 실행 여부를 확인하세요."
            ) from e

def db_ensure_session(session_id: str | None) -> str:
    """session_id가 없으면 새로 발급, 처음 보는 id면 빈 세션으로 생성"""
    if not session_id:
        session_id = str(uuid.uuid4())
    with get_connection() as conn, conn.cursor() as cur:
        # INSERT IGNORE: 이미 있는 세션이면 무시 (PK 중복 에러 방지)
        cur.execute(
            "INSERT IGNORE INTO chat_sessions (session_id, last_recipes) VALUES (%s, %s)",
            (session_id, "[]"),
        )
        conn.commit()
    return session_id

def db_session_exists(session_id: str) -> bool:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM char_sessions WHERE session_id = %s", (session_id,))
        return cur.fetchone() is not None

def db_get_recent_history(session_id: str, limit: int) -> list[dict]:
    """
    최근 limit개 메시지만 시간순으로 조회
    v2에서는 저장할 때 오래된 대화를 지웠지만(trim), 이제는 DB에 전부 남기고
    프롬프트에 넣을 때만 최근 것만 읽어옴 -> 전체 기록은 보존됨
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT role, content FROM(
                SELECT id, role, content FROM chat_messages
                WHERE session_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) AS recent
            ORDER BY id ASC
            """,
            (session_id, limit),
        )
        return [{"role": r["role"], "content": r["content"]} for r in cur.fetchall()]

def db_get_all_history(session_id: str) -> list[dict]:
    """세션의 전체 대화 기록 조회 (히스토리 조회 API용)"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT role, content, created_at FROM chat_messages WHERE session_id = %s ORDER BY id ASC",
            (session_id),
        )
        return [
            {"role": r["role"], "content": r["content"], "created_at": str(r["created_at"])}
            for r in cur.fetchall()
        ]

def db_get_last_recipes(session_id: str) -> list[str]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT last_recipes FROM chat_sessions WHERE session_id = %s", (session_id,))
        row = cur.fetchone()
        return json.loads(row["last_recipes"]) if row else []

def db_save_turn(session_id: str, question: str, answer: str, recipes: list[str]):
    """
    한 턴을 저장: 질문/답변 메시지 2개 + 직전 레시피 갱신
    세 작업을 하나의 트랜잭션으로 묶음 -> 중간에 실패하면 전부 취소 (반쪽 저장 방지)
    JDBC의 setAutoCommit(false) -> commit() / rollback()과 같은 패턴
    """
    with get_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
                    [(session_id, "user", question), (session_id, "assistant", answer)],
                )
                cur.execute(
                    "UPDATE chat_sessions SET last_recipes = %s WHERE session_id = %s",
                    (json.dumps(recipes, ensure_ascii=False), session_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

def db_delete_session(session_id: str) -> bool:
    """세션 삭제 (ON DDELETE CASCADE로 메시지도 함꼐 삭제됨)"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        conn.commit()
        return cur.rowcount > 0

def db_count_sessions() -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM chat_sessions")
        return cur.fetchone()["cnt"]

check_db_connection()


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

def merge_recipes(new_recipes: list[str], previous_recipes: list[str]) -> list[str]:
    """
    직전 턴 레시피 + 이번 검색 결과에서 합침 (중복 제거, 개수 제한)
    이전 레시피를 앞에 둬야 모델이 "그거"를 직전 언급 요리로 해석함
    """
    merged = []
    for recipe in previous_recipes + new_recipes:
        if recipe not in merged:
            merged.append(recipe)
    return merged[:MAX_CONTEXT_RECIPES]

def generate_answer(question, context, history):
    """messages 구성: [system] + [이전 대화들] + [이번 질문 + 레시피 목록]"""
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
        "stored_sessions": db_count_sessions() 
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
    if request.session_id and len(request.session_id) > 36:
        raise HTTPException(status_code=400, detail="session_id 형식이 올바르지 않습니다.")

    session_id = db_ensure_session(request.session_id)
    history = db_get_recent_history(session_id, MAX_TURNS * 2)
    previous_recipes = db_get_last_recipes(session_id)

    # 1) 현재 질문으로 검색
    results = search_with_scores(request.question, top_k=request.top_k)
    retrieved = [r["recipe"] for r in results]

    # 2) 직전 턴 레시피와 합쳐서 맥락 유지
    context_recipes = merge_recipes(retrieved, previous_recipes)
    context = "\n".join(context_recipes)

    # 3) 히스토리 + 레시피 목록으로 답변 생성
    answer = generate_answer(request.question, context, history)

    # 4) 이번 턴 저장 (레시피 목록은 빼고 순수 질문/답변만 히스토리에)
    db_save_turn(session_id, request.question, answer, context_recipes)

    return ChatResponse(
        session_id=session_id,
        question=request.question,
        retrieved_recipes=retrieved,
        context_recipes=context_recipes,
        answer=answer
    )


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str):
    """세션의 전체 대화 기록 조회"""
    if not db_session_exists(session_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 세션입니다.")
    history = db_get_all_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        turns=len(history) // 2,
        history=history
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """세션 삭제 (대화 기록도 함께 삭제)"""
    if not db_delete_session(session_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 세션입니다.")
    return {"deleted": session_id}
