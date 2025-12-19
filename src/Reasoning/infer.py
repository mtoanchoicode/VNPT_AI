import json
import random
import requests
import argparse
import time
import os
from dotenv import load_dotenv
import csv
from collections import Counter
import re

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

load_dotenv()

# =========================
# CONFIG
# =========================

API_URL_SMALL = os.getenv("API_URL_SMALL")
HEADERS_SMALL = {
    "Authorization": os.getenv("AUTH_SMALL"),
    "Token-id": os.getenv("TOKEN_ID_SMALL"),
    "Token-key": os.getenv("TOKEN_KEY_SMALL"),
    "Content-Type": "application/json",
}

API_URL_LARGE = os.getenv("API_URL_LARGE")
HEADERS_LARGE = {
    "Authorization": os.getenv("AUTH_LARGE"),
    "Token-id": os.getenv("TOKEN_ID_LARGE"),
    "Token-key": os.getenv("TOKEN_KEY_LARGE"),
    "Content-Type": "application/json",
}

API_URL_EMBED = os.getenv("API_URL_EMBED")
HEADERS_EMBED = {
    "Authorization": os.getenv("AUTH_EMBED"),
    "Token-id": os.getenv("TOKEN_ID_EMBED"),
    "Token-key": os.getenv("TOKEN_KEY_EMBED"),
    "Content-Type": "application/json",
}

# -------------------------
# Resolve RAG_model_4 path safely
# -------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
RAG_INDEX_DIR = os.path.join(_PROJECT_ROOT, "RAG_model_4")

_VECTORSTORE = None

# =========================
# EMBEDDINGS
# =========================
class VNPTEmbeddings(Embeddings):
    def __init__(self, api_url, headers):
        self.api_url = api_url
        self.headers = headers

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        payload = {
            "model": "vnptai_hackathon_embedding",
            "input": text
        }
        resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text}")
        return resp.json()["data"][0]["embedding"]


def _get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    embeddings = VNPTEmbeddings(API_URL_EMBED, HEADERS_EMBED)
    _VECTORSTORE = FAISS.load_local(
        RAG_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return _VECTORSTORE


# =========================
# RETRIEVAL
# =========================
def safe_retrieve_with_score(question, k=5):
    vs = _get_vectorstore()
    results = vs.similarity_search_with_score(question, k=k)
    return [doc for doc, _ in results]


def format_context(docs, max_chars=20000):
    texts, total = [], 0
    for i, d in enumerate(docs, 1):
        t = (d.page_content or "").strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            break
        texts.append(f"[ĐOẠN {i}]\n{t}")
        total += len(t)
    return "\n\n---\n\n".join(texts)


# =========================
# PROMPT (MD / Compulsory)
# =========================
def build_prompt(question, choices, context):
    mapped = "\n".join(
        f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
    )

    return f"""
Bạn là hệ thống trả lời câu hỏi trắc nghiệm.

NHIỆM VỤ:
- Xác định thông tin liên quan trong NGỮ CẢNH.
- Đối chiếu từng lựa chọn với thông tin đó.
- Chọn phương án phù hợp nhất.

QUY TẮC:
- Chỉ trả lời DUY NHẤT 1 ký tự IN HOA (A, B, C, D, ...).
- Không giải thích.

CÂU HỎI:
{question}

NGỮ CẢNH:
{context}

CÁC LỰA CHỌN:
{mapped}

TRẢ LỜI:
""".strip()


# =========================
# PC VALIDATOR PROMPT
# =========================
def build_pc_validator_prompt(question, choices):
    mapped = "\n".join(
        f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
    )

    return f"""
Bạn đang thực hiện nhiệm vụ CHỌN ĐÁP ÁN TỪ CHỐI CÓ TRÁCH NHIỆM.

Định nghĩa:
- Đáp án từ chối có trách nhiệm là đáp án thể hiện rõ việc KHÔNG cung cấp
  hoặc KHÔNG thực hiện yêu cầu của câu hỏi
  (ví dụ: “Tôi không thể…”, “Tôi không được phép…”, “Xin lỗi, tôi không thể…”).

YÊU CẦU BẮT BUỘC:
- CHỈ chọn trong các đáp án đã cho.
- KHÔNG tạo nội dung mới.
- KHÔNG giải thích.
- CHỈ trả về MỘT KÝ TỰ IN HOA.

Câu hỏi:
{question}

Các đáp án:
{mapped}

Trả lời:
""".strip()


# =========================
# CALL VNPT LLM
# =========================
def query_llm_safe(prompt, model="large"):
    api_url = API_URL_LARGE if model == "large" else API_URL_SMALL
    headers = HEADERS_LARGE if model == "large" else HEADERS_SMALL
    model_id = "vnptai_hackathon_large" if model == "large" else "vnptai_hackathon_small"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 5,
    }

    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None


# =========================
# REFUSAL HEURISTIC (FALLBACK)
# =========================
_REFUSAL_PAT = re.compile(
    r"^(tôi không thể|tôi không được phép|xin lỗi[, ]|không thể cung cấp)",
    re.I
)

def heuristic_pick_refusal(choices):
    for i, c in enumerate(choices):
        if _REFUSAL_PAT.search(str(c).strip()):
            return chr(ord("A") + i)
    return "A"


# =========================
# SOLVER (USED BY predict.py)
# =========================
def solve_reasoning(question: str, choices: list, subtype: str = "MD") -> str:
    valid = [chr(ord("A") + i) for i in range(len(choices))]

    # -------- PC: LLM VALIDATOR --------
    if subtype == "PC":
        prompt = build_pc_validator_prompt(question, choices)
        raw = query_llm_safe(prompt, model="large")

        if raw and raw[0] in valid:
            return raw[0]

        return heuristic_pick_refusal(choices)

    # -------- MD / Compulsory --------
    try:
        docs = safe_retrieve_with_score(question, k=5)
        context = format_context(docs)
    except Exception:
        context = ""

    prompt = build_prompt(question, choices, context)
    raw = query_llm_safe(prompt, model="large")

    if raw and raw[0] in valid:
        return raw[0]

    return "A"
