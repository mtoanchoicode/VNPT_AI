#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import base64
import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests

# Optional: dotenv fallback (safe if not installed / not provided)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


# ------------------ OPTIONAL .ENV LOADING ------------------
def load_env_if_present(dotenv_path: Optional[str] = None) -> None:
    """
    Optional helper. Call this manually only for local runs if desired.
    In Docker/production, prefer passing env vars explicitly.
    """
    if load_dotenv is None:
        return
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
    else:
        load_dotenv()


# ------------------ ENV / API CONFIG ------------------
def _get_env_trim(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def _headers_for(model: str) -> Dict[str, str]:
    if model == "large":
        return {
            "Authorization": _get_env_trim("AUTH_LARGE"),
            "Token-id": _get_env_trim("TOKEN_ID_LARGE"),
            "Token-key": _get_env_trim("TOKEN_KEY_LARGE"),
            "Content-Type": "application/json",
        }
    return {
        "Authorization": _get_env_trim("AUTH_SMALL"),
        "Token-id": _get_env_trim("TOKEN_ID_SMALL"),
        "Token-key": _get_env_trim("TOKEN_KEY_SMALL"),
        "Content-Type": "application/json",
    }


def _endpoint_and_model_id(model: str) -> Tuple[str, str]:
    if model == "large":
        url = _get_env_trim("API_URL_LARGE")
        if not url:
            raise RuntimeError("Missing API_URL_LARGE env var")
        return url, "vnptai_hackathon_large"

    url = _get_env_trim("API_URL_SMALL")
    if not url:
        raise RuntimeError("Missing API_URL_SMALL env var")
    return url, "vnptai_hackathon_small"


def _ensure_auth_headers_present(headers: Dict[str, str], model: str) -> None:
    # Do NOT normalize "Bearer ..." per your requirement.
    # Just fail fast if credentials are missing.
    if not headers.get("Authorization") or not headers.get("Token-id") or not headers.get("Token-key"):
        raise RuntimeError(
            f"Missing auth headers for model={model}. "
            f"Need Authorization, Token-id, Token-key via ENV."
        )


# ------------------ OUTPUT LABELS ------------------
LABEL4_TO_CLASS = {"1": "RAG", "2": "Compulsory", "3": "STEM", "4": "Reasoning"}
VALID_CLASS_NAMES_FINAL = {"RAG", "STEM", "Reasoning"}
VALID_SUBTYPES = {"PC", "MD", "Compulsory", "NA"}


# ------------------ RAG-IN-QUESTION HEURISTIC ONLY ------------------
RAG_IN_QUESTION_PATTERNS = [
    r"\bđoạn thông tin\b",
]


def is_rag_in_question(question: str) -> bool:
    """Detect RAG only when a passage/table/data is INCLUDED in the question."""
    if not question:
        return False
    q_l = question.lower()
    for pat in RAG_IN_QUESTION_PATTERNS:
        if re.search(pat, q_l, flags=re.IGNORECASE | re.MULTILINE):
            return True
    return False


# ------------------ LLM PROMPT (VI ONLY, JSON) ------------------
SYSTEM_PROMPT_VI_JSON = """
Bạn là bộ phân loại câu hỏi tiếng Việt cho benchmark trắc nghiệm.
Chỉ trả về JSON đúng format:
{"label4":"1|2|3|4","subtype":"PC|MD|Compulsory|NA"}

Bạn sẽ được cung cấp:
- QUESTION (câu hỏi)
- CHOICES (các đáp án A/B/C/D)

Quy tắc phân loại:
1 = RAG:
- CHỈ chọn khi trong QUESTION có "Đoạn thông tin" / đoạn văn / bảng / dữ liệu
  ĐƯỢC CUNG CẤP TRỰC TIẾP trong chính QUESTION và yêu cầu dựa vào đó.

2 = Compulsory:
- Chỉ chọn nếu chắc chắn là câu an toàn, kiến thức phổ thông/sách giáo khoa 1–12 ở Việt Nam,
  và "bắt buộc phải trả lời được" (không nhạy cảm, không cần từ chối).
- Nếu không chắc chắn là SGK 1–12 → KHÔNG chọn 2, cân nhắc chọn 4.

3 = STEM:
- Bài toán Toán / Lý / Hóa / Sinh / Tin học / tư duy logic dạng bài tập,
  không cần tra cứu kiến thức ngoài, không có đoạn thông tin kèm theo.

4 = Reasoning:
- Tất cả các trường hợp còn lại (bao gồm xã hội, hành chính, văn hóa, đời sống, tổng hợp, thông tin mới...).
- subtype (chỉ dùng khi label4="4"):
  + PC: những câu mà câu trả lời có trách nhiệm phải là TỪ CHỐI kiểu “Tôi không thể cung cấp…”.
        Ví dụ: xin hướng dẫn tham nhũng/hối lộ, lừa đảo, gian lận, hack/chiếm đoạt,
        bạo lực, chế tạo vũ khí, hướng dẫn gây hại, hoặc hành vi bất hợp pháp/nguy hiểm tương tự.
  + Compulsory: câu an toàn, kiến thức phổ thông/bắt buộc phải trả lời được (tương tự SGK 1–12),
        nhưng nếu bạn không chắc nó thuộc nhãn 2 thì có thể dùng subtype này dưới Reasoning.
  + MD: các câu còn lại.

Nếu label4 != "4" thì subtype = "NA".

Chỉ trả về JSON, không thêm bất kỳ chữ nào khác.
""".strip()


def format_mcq_for_llm(question: str, choices: Any) -> str:
    q = (question or "").strip()
    lines = [f"QUESTION:\n{q}\n", "CHOICES:"]
    if isinstance(choices, list) and choices:
        for i, c in enumerate(choices):
            lines.append(f"{chr(ord('A') + i)}. {str(c).strip()}")
    else:
        lines.append("(none)")
    return "\n".join(lines).strip()


# ------------------ PARSING ------------------
def _normalize_subtype(subtype_raw: str) -> str:
    if not subtype_raw:
        return "NA"
    s = str(subtype_raw).strip()
    up = s.upper()
    if up == "PC":
        return "PC"
    if up == "MD":
        return "MD"
    if up == "NA":
        return "NA"
    if up == "COMPULSORY":
        return "Compulsory"
    return "NA"


def extract_label4_and_subtype(raw: str) -> Tuple[Optional[str], str]:
    """Parse JSON like {"label4":"1|2|3|4","subtype":"PC|MD|Compulsory|NA"}; fallback to regex."""
    if not raw:
        return None, "NA"
    s = raw.strip()

    # 1) strict JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            label4 = str(obj.get("label4", "")).strip()
            subtype = _normalize_subtype(obj.get("subtype", "NA"))
            if label4 not in {"1", "2", "3", "4"}:
                return None, "NA"
            if label4 != "4":
                subtype = "NA"
            return label4, subtype
    except Exception:
        pass

    # 2) regex fallback
    m = re.search(r'"label4"\s*:\s*"([1-4])"', s)
    label4 = m.group(1) if m else None

    m2 = re.search(r'"subtype"\s*:\s*"(PC|MD|COMPULSORY|NA)"', s, flags=re.IGNORECASE)
    subtype = _normalize_subtype(m2.group(1) if m2 else "NA")

    if label4 and label4 != "4":
        subtype = "NA"
    return label4, subtype


# ------------------ VNPT ERROR DECODING ------------------
def _try_decode_vnpt_error_payload(data: Any) -> Optional[Dict[str, Any]]:
    """VNPT sometimes returns {"dataSign":"...","dataBase64":"<base64 json>"}."""
    if not isinstance(data, dict):
        return None
    b64 = data.get("dataBase64")
    if not b64 or not isinstance(b64, str):
        return None
    try:
        raw = base64.b64decode(b64).decode("utf-8", errors="replace")
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else None
    except Exception:
        return None


def _is_safety_or_policy_400(decoded_error: Dict[str, Any]) -> bool:
    """Decide if it's a policy/safety refusal we should NOT retry."""
    try:
        err = decoded_error.get("error") or {}
        code = err.get("code")
        msg = (err.get("message") or "").lower()
        if code != 400:
            return False
        signals = [
            "tôi không thể",
            "không thể cung cấp",
            "không thể hỗ trợ",
            "an toàn",
            "chính sách",
            "policy",
            "vi phạm",
            "bất hợp pháp",
            "từ chối",
            "refuse",
        ]
        return any(s in msg for s in signals)
    except Exception:
        return False


# ------------------ VNPT CHAT COMPLETION ------------------
def vnpt_chat_completion(
    user_content: str,
    model: str,
    system_prompt: str,
    timeout: int = 60,
    max_retries: int = 6,
    seed: int = 42,
) -> str:
    """
    VNPT OpenAI-style chat completions. Returns assistant content.
    Retries on 429/5xx and certain 4xx wrapped errors.
    """
    endpoint, model_id = _endpoint_and_model_id(model)
    headers = _headers_for(model)
    _ensure_auth_headers_present(headers, model=model)

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "top_p": 1,
        "top_k": 20,
        "n": 1,
        "max_completion_tokens": 64,
        "response_format": {"type": "json_object"},
        "seed": seed,
    }

    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)

            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue

            if resp.status_code >= 400:
                try:
                    data = resp.json()
                except Exception:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                decoded = _try_decode_vnpt_error_payload(data)
                if decoded and _is_safety_or_policy_400(decoded):
                    raise ValueError("SAFETY_REFUSAL_400")

                time.sleep(2.0 * (attempt + 1))
                continue

            try:
                data = resp.json()
            except Exception:
                time.sleep(2.0 * (attempt + 1))
                continue

            if isinstance(data, dict) and "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]

            decoded = _try_decode_vnpt_error_payload(data)
            if decoded and _is_safety_or_policy_400(decoded):
                raise ValueError("SAFETY_REFUSAL_400")

            time.sleep(2.0 * (attempt + 1))
            continue

        except ValueError as ve:
            last_err = ve
            raise
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"VNPT API call failed after retries. Last error: {last_err}")


# ------------------ LLM CLASSIFY ONE ------------------
def llm_classify(question: str, choices: Any, model: str) -> Tuple[Optional[str], str, str]:
    """
    Return (label_name_or_none, subtype, status)
    status: "ok" | "safety" | "fail"
    """
    try:
        user_content = format_mcq_for_llm(question, choices)
        raw = vnpt_chat_completion(
            user_content=user_content,
            model=model,
            system_prompt=SYSTEM_PROMPT_VI_JSON,
        )

        label4_digit, subtype = extract_label4_and_subtype(raw)

        if label4_digit and label4_digit in LABEL4_TO_CLASS:
            label_name = LABEL4_TO_CLASS[label4_digit]

            # Merge Compulsory -> Reasoning/Compulsory
            if label_name == "Compulsory":
                label_name = "Reasoning"
                subtype = "Compulsory"

            if label_name != "Reasoning":
                subtype = "NA"
            else:
                if subtype not in {"PC", "MD", "Compulsory"}:
                    subtype = "MD"

            return label_name, subtype, "ok"

        return None, "NA", "fail"

    except ValueError as ve:
        if "SAFETY_REFUSAL" in str(ve):
            return None, "NA", "safety"
        return None, "NA", "fail"

    except Exception:
        return None, "NA", "fail"


def classify_one(
    question: str,
    choices: Any,
    model: str = "large",
) -> Tuple[str, str]:
    """
    Final router used by predict.py

    Returns:
      (label, subtype)
        label in {"RAG","STEM","Reasoning"}
        subtype in {"PC","MD","Compulsory","NA"}
    """
    q = question or ""

    # 1) RAG-in-question heuristic ONLY (NO LLM)
    if is_rag_in_question(q):
        return "RAG", "NA"

    # 2) LLM classify
    label_name, subtype, status = llm_classify(q, choices, model=model)

    if label_name:
        # Enforce: RAG must be in-question; otherwise convert to Reasoning/MD
        if label_name == "RAG" and not is_rag_in_question(q):
            return "Reasoning", "MD"

        # subtype rules
        if label_name != "Reasoning":
            return label_name, "NA"

        if subtype not in {"PC", "MD", "Compulsory"}:
            subtype = "MD"
        return "Reasoning", subtype

    # 3) Fallbacks
    if status == "safety":
        return "Reasoning", "PC"
    return "Reasoning", "MD"
