import json
import openai
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

load_dotenv()
# ================== CONFIG ==================
MODEL = "gpt-3.5-turbo"                       # or "gpt-4-turbo" / "gpt-4o-mini" if you want cheaper
INPUT_FILE = "../data/val.json"                    # your input file
OUTPUT_FILE = "../data/val_with_class.json"        # output file

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ============================================

SYSTEM_PROMPT = """Bạn là chuyên gia đánh giá benchmark LLM tiếng Việt. 
Hãy phân loại câu hỏi sau vào đúng MỘT trong 5 nhóm sau (chỉ trả về đúng 1 từ, không giải thích):

- Precision-Critical → câu hỏi rất khó, yêu cầu suy luận chính xác cao, thường là khoa học nâng cao, y khoa, luật, logic phức tạp
- Compulsory        → kiến thức phổ thông, từ lớp 1 đến lớp 12 (Toán, Lý, Hóa, Sinh, Văn, Sử, Địa, GDCD, tiếng Anh cơ bản…)
- RAG                → câu hỏi cần tra cứu thông tin cụ thể từ đoạn văn dài được cung cấp trong chính câu hỏi
- STEM               → câu hỏi khoa học tự nhiên thuần túy (Toán, Vật lý, Hóa học, Sinh học, Tin học) ở mức học sinh trở lên, không quá phức tạp
- Multi-Domain      → các lĩnh vực khác, môn xã hội nâng cao, lịch sử, địa lý, văn học, tôn giáo, nghệ thuật, sự kiện cụ thể…

Chỉ trả về đúng 1 trong 5 từ trên, không thêm bất kỳ ký tự nào khác."""

def classify_question(question_obj):
    question = question_obj["question"]
    choices = "\n".join(question_obj.get("choices", []))
    
    user_content = f"Câu hỏi:\n{question}\n\nCác lựa chọn:\n{choices}"

    try:
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=10
        )
        cls = response.choices[0].message.content.strip()
        
        # Đảm bảo đúng 5 class hợp lệ
        valid_classes = {"Precision-Critical", "Compulsory", "RAG", "STEM", "Multi-Domain"}
        if cls not in valid_classes:
            cls = "Multi-Domain"  # fallback
        
        return cls
        
    except Exception as e:
        print(f"Error on qid {question_obj['qid']}: {e}")
        return "Multi-Domain"  # fallback khi lỗi

# ================== MAIN ==================
print("Loading data...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions. Classifying...")

# Write proper JSON array with commas
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("[\n")                                      # opening bracket
    for i, item in enumerate(tqdm(data, desc="Classifying")):
        item["class"] = classify_question(item)
        json_line = json.dumps(item, ensure_ascii=False, indent=2)
        if i > 0:
            f.write(",\n")                              # comma before every item except first
        f.write("  " + json_line)                       # 2-space indent for beauty
    f.write("\n]")                                      # closing bracket

print(f"Done! Saved valid JSON array → {OUTPUT_FILE}")