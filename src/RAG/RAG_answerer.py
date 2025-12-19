import os
import re
from typing import List, Tuple
import math
import requests
import argparse


API_URL_EMBED = os.getenv("API_URL_EMBED")
AUTH_EMBED = os.getenv("AUTH_EMBED")
TOKEN_ID_EMBED = os.getenv("TOKEN_ID_EMBED")
TOKEN_KEY_EMBED = os.getenv("TOKEN_KEY_EMBED")

# SMALL LLM
API_URL_SMALL = os.getenv("API_URL_SMALL")
HEADERS_SMALL = {
    "Authorization": os.getenv("AUTH_SMALL"),
    "Token-id": os.getenv("TOKEN_ID_SMALL"),
    "Token-key": os.getenv("TOKEN_KEY_SMALL"),
    "Content-Type": "application/json",
}

# LARGE LLM
API_URL_LARGE = os.getenv("API_URL_LARGE")
HEADERS_LARGE = {
    "Authorization": os.getenv("AUTH_LARGE"),
    "Token-id": os.getenv("TOKEN_ID_LARGE"),
    "Token-key": os.getenv("TOKEN_KEY_LARGE"),
    "Content-Type": "application/json",
}


# CALL VNPT LLM
def query_llm(prompt, model="large"):
    if model == "small":
        payload = {
            "model": "vnptai_hackathon_small",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "max_completion_tokens": 1000,
            "n": 1
        }
        resp = requests.post(API_URL_SMALL, headers=HEADERS_SMALL, json=payload)
    else:
        payload = {
            "model": "vnptai_hackathon_large",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "max_completion_tokens": 1000,
            "n": 1
        }
        resp = requests.post(API_URL_LARGE, headers=HEADERS_LARGE, json=payload)

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip()
    except Exception:
        print("ERROR:", resp.text)
        return None


def split_qna(text: str) -> Tuple[str, str]:
    """
    Split the information from the question.
    Assumption: last line is question, above is context.
    """
    if not text:
        return "", ""

    t = text.replace("\r\n", "\n").replace("\r", "\n").rstrip()

    idx = t.rfind("\n")
    if idx == -1:
        return "", t.strip()

    context = t[:idx].strip()
    qna = t[idx + 1:].strip()

    return context, qna


def chunk_paragraph(paragraph: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
    """
    Splits text into chunks of `chunk_size` words with `overlap` words.
    """
    if not paragraph or not paragraph.strip():
        return []

    words = paragraph.split()

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        if i + chunk_size >= len(words):
            break

    return chunks


def create_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Returns embeddings for each chunk using VNPT AI embedding API.
    Calls the API once per chunk.
    """
    if not chunks:
        return []

    if not all([API_URL_EMBED, AUTH_EMBED, TOKEN_ID_EMBED, TOKEN_KEY_EMBED]):
        raise EnvironmentError("Missing embedding environment variables")

    # keep as you had (embedding auth normalization is OK to keep)
    auth_value = AUTH_EMBED.strip()
    if not auth_value.lower().startswith("bearer "):
        auth_value = f"Bearer {auth_value}"

    headers = {
        "Authorization": auth_value,
        "Token-id": TOKEN_ID_EMBED,
        "Token-key": TOKEN_KEY_EMBED,
        "Content-Type": "application/json",
    }

    embeddings: List[List[float]] = []

    for idx, chunk in enumerate(chunks):
        text = (chunk or "").strip()
        if not text:
            raise ValueError(f"Empty chunk at index {idx}")

        payload = {
            "model": "vnptai_hackathon_embedding",
            "input": text,
            "encoding_format": "float",
        }

        resp = requests.post(API_URL_EMBED, headers=headers, json=payload, timeout=60)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API error at index {idx}: "
                f"HTTP {resp.status_code} - {resp.text}"
            )

        data = resp.json()

        try:
            vec = data["data"][0]["embedding"]
        except Exception:
            raise RuntimeError(f"Unexpected embedding response at index {idx}: {data}")

        embeddings.append(vec)

    return embeddings


# ============================
# Cosine similarity and top-k retrieval
def l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_sim(a: List[float], b: List[float]) -> float:
    na, nb = l2_norm(a), l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def topk_retrieve(
    question_emb: List[float],
    chunk_embs: List[List[float]],
    chunks: List[str],
    k: int = 5
) -> List[Tuple[int, float, str]]:
    scored = []
    for i, emb in enumerate(chunk_embs):
        s = cosine_sim(question_emb, emb)
        scored.append((i, s, chunks[i]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ============================
# Prompts and API calls
def build_RAG_prompt(question: str, context, choices):
    formatted_choices = [f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)]
    formatted_choices = "\n".join(formatted_choices)

    prompt = f"""
Bạn là chuyên gia đọc hiểu và suy luận đáp án từ đoạn thông tin được cung cấp.
Nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm dựa trên thông tin đó.
Nếu không có đủ thông tin, hãy chọn đáp án phù hợp nhất với đoạn thông tin.

---

Hướng dẫn xử lý:
- Đọc từng câu trong "Đoạn thông tin" một cách tuần tự.
- Với mỗi câu, hãy hiểu ngữ cảnh của nó trong mối liên hệ với câu trước và sau nó.
- Đối chiếu câu hỏi với các chi tiết vừa đọc để tìm bằng chứng chính xác.
- Sau khi phân tích, hãy đưa ra đáp án cuối cùng.
- [QUAN TRỌNG] Phân tích ngắn gọn, không lặp lại, tối đa 250 cho PHÂN TÍCH.

Định dạng trả về bắt buộc (bạn phải tuân thủ khuôn mẫu này):
[PHÂN TÍCH]
(Viết quá trình đọc hiểu và suy luận từng bước tại đây, giới hạn suy luận dưới 250 từ)

[ĐÁP ÁN]
(Duy nhất một chữ cái: A, B, C hoặc D, không giải thích thêm)

---

Câu hỏi:
{question}

Các lựa chọn:
{formatted_choices}

Đoạn thông tin:
{context}
""".strip()
    return prompt


def parse_answer(llm_response: str):
    match = re.search(r'\[ĐÁP ÁN\]\s*([A-Z])', llm_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "A"

# ==============================
# Adapter for predict.py
# ==============================

def solve_rag(question: str, choices: list) -> str:
    """
    Solve ONE RAG question.
    Return: "A" | "B" | "C" | "D"
    """
    context, q = split_qna(question)

    if not context.strip():
        prompt = build_RAG_prompt(q, "", choices)
        raw = query_llm(prompt, model="large")
        return parse_answer(raw or "") or "A"

    chunks = chunk_paragraph(context)[:40]

    chunk_embs = create_embeddings(chunks)
    q_emb = create_embeddings([q])[0]

    hits = topk_retrieve(q_emb, chunk_embs, chunks, k=3)
    top_texts = [txt for _, _, txt in hits]
    compact_context = "\n\n".join(top_texts)

    prompt = build_RAG_prompt(q, compact_context, choices)
    raw = query_llm(prompt, model="large")
    return parse_answer(raw or "") or "A"