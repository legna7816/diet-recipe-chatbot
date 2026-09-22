# RAG API Dockerfile
# 빌드: docker build -t rag-api .
# 실행: docker run -p 8000:8000 rag-api

# 베이즈 이미지 3.13 슬림 버전
# slim = 불필요한 패키지를 뺀 경량 이미지
FROM python:3.13-slim

# 작업 디렉토리 설정 (컨테이너 안에서의 경로)
WORKDIR /app

# 의존성 설치
# 중요: requirments.txt만 먼저 복사하는 이유는 Docker 레이어 캐싱 때문
# 코드만 수정했을 때 패키지 재설치를 건너뛸 수 있어 빌드가 훨씬 빨라짐
COPY requirements.txt .

# --no-cache-dir: pip 캐시를 남지지 않아 이미지 크기 감소
RUN pip install --no-cache-dir -r requirements.txt

# 모델 사전 다운로드 (ML d앱 특유의 최적화)
# 이 단계가 없으면 컨테이너를 실행할 때마다 모델을 새로 다운로드함
# 컨테이너는 종료되면 내부 파일이 사라지기에 캐시가 유지되지 않음
# 빌드 시점에 따라 미리 받아서 이미지 안에 포함시키면 실행이 즉시 시작됨
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('jhgan/ko-sroberta-multitask')"

RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# 애플리케이션 코드 복사
# 코드는 자주 바뀌므로 마지막에 복사 (앞 레이어 캐시를 최대한 활용)
COPY main.py .
# 컨테이너가 사용할 포트 명시 (문서화 목적, 실제 개방은 docker run -p)
EXPOSE 8000

# 실행 명령
# --host 0.0.0.0 필수
# 기본값인 127.0.0.1로 두면 컨테이너 내부에서만 접속 가능해져
# 호스트(내 PC)에서 접속이 안 됨
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]