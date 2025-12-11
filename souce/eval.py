import json
import random
import requests
import argparse
import time
import os
from dotenv import load_dotenv
load_dotenv()

# CONFIG

# SMALL LLM
# 1000 API Request / ngày - 60 request / tiếng
# Context length = 32k
# Input length = 28k
API_URL_SMALL = os.getenv("API_URL_SMALL")
HEADERS_SMALL = {
    "Authorization": os.getenv("AUTH_SMALL"),
    "Token-id": os.getenv("TOKEN_ID_SMALL"),
    "Token-key": os.getenv("TOKEN_KEY_SMALL"),
    "Content-Type": "application/json",
}

# LARGE LLM
# 500 API Request Small / ngày - 40 request / tiếng
# Context length = 22k
# Input length = 18k
API_URL_LARGE = os.getenv("API_URL_LARGE")
HEADERS_LARGE = {
    "Authorization": os.getenv("AUTH_LARGE"),
    "Token-id": os.getenv("TOKEN_ID_LARGE"),
    "Token-key": os.getenv("TOKEN_KEY_LARGE"),
    "Content-Type": "application/json",
}


# PROMPT TEMPLATE
def build_prompt(question, choices):
    """
    Build a prompt that forces LLM to output only A/B/C/D.
    """
    mapped = "\n".join([
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(choices)
    ])

    prompt = f"""
Bạn là một hệ thống trả lời trắc nghiệm.

Câu hỏi:
{question}

Các lựa chọn:
{mapped}

Nhiệm vụ:
- Phân tích thật ngắn gọn.
- Chỉ được trả lời duy nhất 1 ký tự như A, B, C.
- KHÔNG được giải thích.
- KHÔNG được viết thêm nội dung khác.

Trả lời:
""".strip()

    return prompt


# ============================
# CALL VNPT LLM
def query_llm(prompt, model = "small"):
    if model == "small":
        payload = {
            "model": "vnptai_hackathon_small",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,   # deterministic for evaluation
            "top_p": 1.0,
            "top_k": 20,
            "max_completion_tokens": 5,
            "n": 1
        }

        resp = requests.post(API_URL_SMALL, headers=HEADERS_SMALL, json=payload)
    else:
        payload = {
            "model": "vnptai_hackathon_large",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,   # deterministic for evaluation
            "top_p": 1.0,
            "top_k": 20,
            "max_completion_tokens": 5,
            "n": 1
        }

        resp = requests.post(API_URL_LARGE, headers=HEADERS_LARGE, json=payload)
    try:
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip()
    except Exception:
        print("ERROR:", resp.text)
        return None


# ============================
# MAIN EVALUATION LOOP
# ============================

def evaluate(dataset_path, model, limit):
    import json, time

    # Output file
    output_file = "eval_results_2.txt"
    f_out = open(output_file, "w", encoding="utf-8")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit != -1:
        data = data[:limit]

    correct = 0
    total = len(data)
    latencies = []   # store inference time

    print(f"\nEvaluating {total} samples...\n")
    f_out.write(f"Evaluating {total} samples...\n\n")

    for i, item in enumerate(data, 1):
        qid = item["qid"]
        question = item["question"]
        choices = item["choices"]
        gt_answer = item["answer"]  # A/B/C/D
        ques_class = item["class"]

        prompt = build_prompt(question, choices)

        start = time.time()
        pred = query_llm(prompt, model)
        end = time.time()

        latency = end - start
        latencies.append(latency)

        # Clean prediction (only keep first letter)
        if pred:
            pred = pred.strip().upper()[0]

        # Check correctness
        is_correct = (pred == gt_answer)
        if is_correct:
            correct += 1

        line = (f"[{i}/{total}] {qid} | GT: {gt_answer} | LLM: {pred} | "
                f"{'✔' if is_correct else '✘'} | Class: {ques_class} | "
                f"Time: {latency:.3f}s")

        print(line)
        f_out.write(line + "\n")

    accuracy = (correct / total) * 100
    avg_latency = sum(latencies) / len(latencies)

    summary = (
        "\n=============================\n"
        f"Accuracy: {accuracy:.2f}% ({correct}/{total})\n"
        f"Avg Inference Time: {avg_latency:.3f} seconds\n"
        f"Min/Max Latency: {min(latencies):.3f}s / {max(latencies):.3f}s\n"
        "=============================\n"
    )

    print(summary)
    f_out.write(summary)

    f_out.close()
    print(f"Results saved to {output_file}")


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=-1,
                        help="Number of samples to evaluate (1,2,5,10 or -1 for ALL)")
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--file", type=str, default="../data/val_with_class.json")
    args = parser.parse_args()

    evaluate(args.file, args.model, args.limit)



#   {
#     "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6IjY1ZmNhZjI3LTI1Y2ItNGY5ZC1iMTgwLWUyMzM5NjU4ZDZjYyIsInN1YiI6IjE1N2YyODllLWQzNzMtMTFmMC1hNzY5LWIzODRjNTI3ODgzZCIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJ0b2FudHIud29ya0BnbWFpbC5jb20iLCJzY29wZSI6WyJyZWFkIl0sImlzcyI6Imh0dHBzOi8vbG9jYWxob3N0IiwibmFtZSI6InRvYW50ci53b3JrQGdtYWlsLmNvbSIsInV1aWRfYWNjb3VudCI6IjE1N2YyODllLWQzNzMtMTFmMC1hNzY5LWIzODRjNTI3ODgzZCIsImF1dGhvcml0aWVzIjpbIlVTRVIiLCJUUkFDS18yIl0sImp0aSI6IjEyODI5ZTlhLTdhZmYtNDg4MC1iYmQwLWNhYTIyZTYxMTAyYSIsImNsaWVudF9pZCI6ImFkbWluYXBwIn0.VgxhPvPGBmtY5X5DuJRps1jNGfkKBzrEg42hPIcy1MSKySR3d-CxFiXTqIWqZZB6klMWjrMNjA-8i0MsTj_YqeoZ029JrXQC5zER1HIfVmFLwZ3TsGA6cNwyDgVIxu6dFv7moxO4_O5AvPZy7XVhhdCLMuQZlKQ6de9IHrAnprUa2bBQ1Qv9AeMzV-asjVEsjVcEo4y07AnJj0PEYzX234k8B3gKh3lz7zhRJD_4F4cD-Em3a6najwOltzUy2zqMVbCBGE5M7KVJGBzMU1moG_DMy1PzXa329_128imjf9nh_LLu9UGDFI_3APaC5tjsJdbhjJ_jRBNnwTHvdIKPnQ",
#     "tokenKey": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAIbL+oX+qRdtDoP1Wd81pJena4/kkFlJruXeUDxqTt35NPqTM56rHDy4B8EXagWCnX6yk9nTQHHMlcN5+E+pQtkCAwEAAQ==",
#     "llmApiName": "LLM embedings",
#     "tokenId": "457df239-3796-0839-e063-63199f0a802b"
#   },