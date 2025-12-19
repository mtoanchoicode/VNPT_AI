import json
import re
import time
import csv
import requests
import os
# from dotenv import load_dotenv
from tqdm import tqdm

# =====================
# LOAD ENV
# =====================
# load_dotenv()

API_URL_SMALL = os.getenv("API_URL_SMALL")
HEADERS_SMALL = {
    "Authorization": os.getenv("AUTH_SMALL"),
    "Token-id": os.getenv("TOKEN_ID_SMALL"),
    "Token-key": os.getenv("TOKEN_KEY_SMALL"),
    "Content-Type": "application/json",
}
WAIT_TIME_ON_QUOTA = 60 * 60
MODEL_NAME = "vnptai_hackathon_small"

# =====================
# FILE CONFIG
# =====================
INFERENCE_TIME_FILE = "vnpt_small_inference_time.csv"
THINKING_FILE = "vnpt_small_explanations.json"
ANSWER_FILE = "vnpt_small_answers.json"
CSV_FILE = "vnpt_small_answers.csv"
PROGRESS_FILE = "vnpt_small_progress.txt"

# ====================================
# PROMT ENGINEERING (CHAIN-OF-THOUGHT)
# ====================================
def build_cot_prompt(question, choices):
    mapped_choices = "\n".join([
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(choices)
    ])

    prompt = f"""
Bạn là một chuyên gia giải đề thi STEM (Khoa học, Công nghệ, Kỹ thuật, Toán học) với độ chính xác tuyệt đối.

NHIỆM VỤ:
Giải quyết câu hỏi trắc nghiệm dưới đây bằng phương pháp suy luận từng bước (Chain-of-Thought).

QUY TẮC BẮT BUỘC:
1. SUY LUẬN: Phân tích đề bài, xác định công thức hoặc lý thuyết liên quan.
2. TÍNH TOÁN: Nếu có số liệu, hãy viết phép tính rõ ràng, thay số từng bước. Không được làm tắt.
3. KẾT LUẬN: Sau khi suy luận xong, bắt buộc phải chốt đáp án ở dòng cuối cùng theo định dạng:
### ANSWER: X
(Trong đó X là ký tự A, B, C, D, hoặc các ký tự khác tương ứng với đáp án đúng).

--------------------------------------------------
VÍ DỤ MẪU (Hãy làm theo format này):

CÂU HỎI:
Một vật rơi tự do từ độ cao h = 20m, lấy g = 10m/s². Thời gian rơi của vật là:
A. 1s
B. 2s
C. 3s
D. 4s

SUY LUẬN:
- Đây là bài toán rơi tự do.
- Công thức tính thời gian rơi: t = sqrt(2h / g).
- Thay số vào công thức:
  h = 20m
  g = 10m/s²
  t = sqrt(2 * 20 / 10) = sqrt(40 / 10) = sqrt(4) = 2 (giây).
- So sánh với các lựa chọn:
  A. 1s (Sai)
  B. 2s (Đúng)
  C. 3s (Sai)
  D. 4s (Sai)
- Vậy đáp án đúng là B.

### ANSWER: B
--------------------------------------------------

BÂY GIỜ LÀ CÂU HỎI CỦA BẠN:

CÂU HỎI:
{question}

CÁC LỰA CHỌN:
{mapped_choices}

SUY LUẬN:
""".strip()
    return prompt

# =====================
# API CALL
# =====================
def query_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 2048,
    }

    r = requests.post(
        API_URL_SMALL,
        headers=HEADERS_SMALL,
        json=payload,
        timeout=300
    )

    if r.status_code in (429, 403):
        raise RuntimeError("RATE_LIMIT_REACHED")

    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]

    raise RuntimeError(f"API Error {r.status_code}: {r.text}")


# =====================
# EXTRACT ANSWER
# =====================
def extract_answer(llm_output):
    if not llm_output:
        return "N/A"
    
    match = re.search(r"### ANSWER:\s*([A-Z])", llm_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    fallback_match = re.search(r"(?:Answer|Đáp án|Choice|Lựa chọn)[:\s]*([A-Z])", llm_output, re.IGNORECASE)
    if fallback_match:
        return fallback_match.group(1).upper()
    
    return "FAIL"


# =====================
# MAIN WORKER FUNCTION
# =====================
def run_stem_worker(input_file: str):
    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load checkpoint
    start_idx = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            start_idx = int(f.read().strip())

    thinking_results = []
    answer_results = []
    inference_times = []

    if os.path.exists(INFERENCE_TIME_FILE):
        with open(INFERENCE_TIME_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            inference_times = list(reader)

    if os.path.exists(THINKING_FILE):
        with open(THINKING_FILE, "r", encoding="utf-8") as f:
            thinking_results = json.load(f)

    if os.path.exists(ANSWER_FILE):
        with open(ANSWER_FILE, "r", encoding="utf-8") as f:
            answer_results = json.load(f)

    i = start_idx

    while i < len(data):
        item = data[i]
        qid = item["qid"]

        try:
            prompt = build_cot_prompt(item["question"], item["choices"])
            start_time = time.perf_counter()
            content = query_llm(prompt)
            end_time = time.perf_counter()

            inference_time = round(end_time - start_time, 4)
            answer = extract_answer(content)

            thinking_results.append({
                "qid": qid,
                "question": item["question"],
                "choices": item["choices"],
                "explanation": content
            })

            inference_times.append({
                "qid": qid,
                "inference_time_sec": inference_time
            })

            answer_results.append({
                "qid": qid,
                "answer": answer
            })

            # SAVE JSON
            with open(THINKING_FILE, "w", encoding="utf-8") as f:
                json.dump(thinking_results, f, ensure_ascii=False, indent=2)

            with open(ANSWER_FILE, "w", encoding="utf-8") as f:
                json.dump(answer_results, f, ensure_ascii=False, indent=2)

            # SAVE CSV
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
                writer.writeheader()
                writer.writerows(answer_results)

            with open(INFERENCE_TIME_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["qid", "inference_time_sec"]
                )
                writer.writeheader()
                writer.writerows(inference_times)

            # SAVE CHECKPOINT
            with open(PROGRESS_FILE, "w") as f:
                f.write(str(i + 1))

            i += 1

        except RuntimeError as e:
            if "Rate limit exceed" in str(e):
                print("Rate limit reached. Sleeping 61 minutes...")
                time.sleep(WAIT_TIME_ON_QUOTA)
                print("Resume working...")
            else:
                print(f"[ERROR] {qid}: {e}")
                i += 1

    print("FINISHING")

# ==============================
# Adapter for predict.py (Docker-safe)
# ==============================
def solve_stem(question: str, choices: list) -> str:
    """
    Solve ONE STEM question.
    Return: "A" | "B" | "C" | "D"
    - Docker-safe: không để crash predict.py nếu gặp RATE_LIMIT/HTTP lỗi
    """
    prompt = build_cot_prompt(question, choices)
    try:
        raw = query_llm(prompt)
    except Exception:
        # fallback an toàn để pipeline không sập
        return "A"
    return extract_answer(raw) or "A"
